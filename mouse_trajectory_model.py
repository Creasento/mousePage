import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


GROUP_KEYS = ("subject", "subfolder", "condition_index", "trial_number", "segment_index")
COND_KEYS = ("A", "W", "ID", "duration")


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except ValueError:
        return default


def read_trials(csv_path):
    groups = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error", "false").lower() == "true":
                continue
            key = tuple(row.get(k, "") for k in GROUP_KEYS)
            groups[key].append(row)

    trials = []
    for rows in groups.values():
        rows.sort(key=lambda r: _safe_float(r.get("point_index")))
        points = np.array(
            [[_safe_float(r.get("x_norm")), _safe_float(r.get("y_norm"))] for r in rows],
            dtype=np.float32,
        )
        if len(points) < 4:
            continue

        first = rows[0]
        duration = _safe_float(first.get("duration"), default=len(points) - 1)
        if duration <= 0:
            duration = len(points) - 1

        cond = np.array(
            [
                _safe_float(first.get("A")),
                _safe_float(first.get("W")),
                _safe_float(first.get("ID")),
                duration,
            ],
            dtype=np.float32,
        )
        trials.append((cond, points))

    if not trials:
        raise ValueError(f"No usable trials found in {csv_path}")

    return trials


def resample_points(points, seq_len):
    old_t = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.interp(new_t, old_t, points[:, 0])
    y = np.interp(new_t, old_t, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


class Standardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float32)
        return cls(values.mean(axis=0), values.std(axis=0))

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


class TrajectoryDataset(Dataset):
    def __init__(self, trials, seq_len, cond_scaler=None):
        self.seq_len = seq_len
        self.conds = np.stack([cond for cond, _ in trials])
        self.cond_scaler = cond_scaler or Standardizer.fit(self.conds)
        self.conds = self.cond_scaler.transform(self.conds)
        self.points = np.stack([resample_points(points, seq_len) for _, points in trials])

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.conds[index]),
            torch.from_numpy(self.points[index]),
        )


class TrajectoryCVAE(nn.Module):
    def __init__(self, cond_dim=4, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(input_size=2 + cond_dim, hidden_size=hidden_dim, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_in = nn.Linear(cond_dim + latent_dim + 1, hidden_dim)
        self.decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2)

    def encode(self, cond, traj):
        cond_seq = cond[:, None, :].expand(-1, traj.size(1), -1)
        _, hidden = self.encoder(torch.cat([traj, cond_seq], dim=-1))
        hidden = hidden[-1]
        return self.to_mu(hidden), self.to_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, cond, z, seq_len):
        t = torch.linspace(0.0, 1.0, seq_len, device=cond.device)
        t = t[None, :, None].expand(cond.size(0), -1, -1)
        cond_seq = cond[:, None, :].expand(-1, seq_len, -1)
        z_seq = z[:, None, :].expand(-1, seq_len, -1)
        x = self.decoder_in(torch.cat([cond_seq, z_seq, t], dim=-1))
        decoded, _ = self.decoder(torch.tanh(x))
        traj = self.out(decoded)

        start = torch.zeros_like(traj[:, :1, :])
        end = torch.tensor([1.0, 0.0], device=cond.device).view(1, 1, 2).expand(cond.size(0), -1, -1)
        return torch.cat([start, traj[:, 1:-1, :], end], dim=1)

    def forward(self, cond, traj):
        mu, logvar = self.encode(cond, traj)
        z = self.reparameterize(mu, logvar)
        return self.decode(cond, z, traj.size(1)), mu, logvar


def smoothness_loss(traj):
    velocity = traj[:, 1:] - traj[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return acceleration.pow(2).mean()


def velocity_loss(pred, target):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    return F.mse_loss(pred_velocity, target_velocity)


def acceleration_loss(pred, target):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    return F.mse_loss(pred_acceleration, target_acceleration)


def _mean_magnitude(values):
    return torch.linalg.norm(values, dim=-1).mean(dim=1)


def _relative_stat_loss(pred_values, target_values):
    pred_mean = pred_values.mean()
    target_mean = target_values.mean()
    pred_std = pred_values.std(unbiased=False)
    target_std = target_values.std(unbiased=False)
    eps = 1e-6
    mean_gap = ((pred_mean - target_mean) / (target_mean.abs() + eps)).pow(2)
    std_gap = ((pred_std - target_std) / (target_std.abs() + eps)).pow(2)
    return mean_gap + std_gap


def acceleration_stat_loss(pred, target):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    return _relative_stat_loss(
        _mean_magnitude(pred_acceleration),
        _mean_magnitude(target_acceleration),
    )


def jerk_stat_loss(pred, target):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    pred_jerk = pred_acceleration[:, 1:] - pred_acceleration[:, :-1]
    target_jerk = target_acceleration[:, 1:] - target_acceleration[:, :-1]
    return _relative_stat_loss(
        _mean_magnitude(pred_jerk),
        _mean_magnitude(target_jerk),
    )


def deviation_stat_loss(pred, target):
    return _relative_stat_loss(
        pred[:, :, 1].abs().mean(dim=1),
        target[:, :, 1].abs().mean(dim=1),
    )


def path_length_stat_loss(pred, target):
    pred_steps = pred[:, 1:] - pred[:, :-1]
    target_steps = target[:, 1:] - target[:, :-1]
    return _relative_stat_loss(
        torch.linalg.norm(pred_steps, dim=-1).sum(dim=1),
        torch.linalg.norm(target_steps, dim=-1).sum(dim=1),
    )


def peak_velocity_stat_loss(pred, target):
    pred_steps = pred[:, 1:] - pred[:, :-1]
    target_steps = target[:, 1:] - target[:, :-1]
    return _relative_stat_loss(
        torch.linalg.norm(pred_steps, dim=-1).max(dim=1).values,
        torch.linalg.norm(target_steps, dim=-1).max(dim=1).values,
    )


def _late_slice(values, ratio):
    start = int(values.size(1) * (1.0 - ratio))
    return values[:, max(0, start) :]


def late_acceleration_stat_loss(pred, target, ratio=0.3):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    return _relative_stat_loss(
        torch.linalg.norm(_late_slice(pred_acceleration, ratio), dim=-1).mean(dim=1),
        torch.linalg.norm(_late_slice(target_acceleration, ratio), dim=-1).mean(dim=1),
    )


def late_jerk_stat_loss(pred, target, ratio=0.3):
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    pred_jerk = pred_acceleration[:, 1:] - pred_acceleration[:, :-1]
    target_jerk = target_acceleration[:, 1:] - target_acceleration[:, :-1]
    return _relative_stat_loss(
        torch.linalg.norm(_late_slice(pred_jerk, ratio), dim=-1).mean(dim=1),
        torch.linalg.norm(_late_slice(target_jerk, ratio), dim=-1).mean(dim=1),
    )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trials = read_trials(args.csv)
    if len(trials) < 2:
        raise ValueError("At least 2 trials are required for train/validation split")

    dataset = TrajectoryDataset(trials, args.seq_len)
    val_count = max(1, int(len(dataset) * args.val_ratio))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrajectoryCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for cond, traj in train_loader:
            cond = cond.to(device)
            traj = traj.to(device)
            pred, mu, logvar = model(cond, traj)
            recon = F.mse_loss(pred, traj)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vel = velocity_loss(pred, traj)
            acc = acceleration_loss(pred, traj)
            acc_stat = acceleration_stat_loss(pred, traj)
            jerk_stat = jerk_stat_loss(pred, traj)
            dev_stat = deviation_stat_loss(pred, traj)
            path_stat = path_length_stat_loss(pred, traj)
            peak_vel_stat = peak_velocity_stat_loss(pred, traj)
            late_acc_stat = late_acceleration_stat_loss(pred, traj, args.late_ratio)
            late_jerk_stat = late_jerk_stat_loss(pred, traj, args.late_ratio)
            loss = (
                recon
                + args.beta * kld
                + args.smooth_weight * smoothness_loss(pred)
                + args.velocity_weight * vel
                + args.acceleration_weight * acc
                + args.acceleration_stat_weight * acc_stat
                + args.jerk_stat_weight * jerk_stat
                + args.deviation_stat_weight * dev_stat
                + args.path_length_stat_weight * path_stat
                + args.peak_velocity_stat_weight * peak_vel_stat
                + args.late_acceleration_stat_weight * late_acc_stat
                + args.late_jerk_stat_weight * late_jerk_stat
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * cond.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, traj in val_loader:
                cond = cond.to(device)
                traj = traj.to(device)
                pred, mu, logvar = model(cond, traj)
                recon = F.mse_loss(pred, traj)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vel = velocity_loss(pred, traj)
                acc = acceleration_loss(pred, traj)
                acc_stat = acceleration_stat_loss(pred, traj)
                jerk_stat = jerk_stat_loss(pred, traj)
                dev_stat = deviation_stat_loss(pred, traj)
                path_stat = path_length_stat_loss(pred, traj)
                peak_vel_stat = peak_velocity_stat_loss(pred, traj)
                late_acc_stat = late_acceleration_stat_loss(pred, traj, args.late_ratio)
                late_jerk_stat = late_jerk_stat_loss(pred, traj, args.late_ratio)
                loss = (
                    recon
                    + args.beta * kld
                    + args.smooth_weight * smoothness_loss(pred)
                    + args.velocity_weight * vel
                    + args.acceleration_weight * acc
                    + args.acceleration_stat_weight * acc_stat
                    + args.jerk_stat_weight * jerk_stat
                    + args.deviation_stat_weight * dev_stat
                    + args.path_length_stat_weight * path_stat
                    + args.peak_velocity_stat_weight * peak_vel_stat
                    + args.late_acceleration_stat_weight * late_acc_stat
                    + args.late_jerk_stat_weight * late_jerk_stat
                )
                val_loss += loss.item() * cond.size(0)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d} "
                f"train={train_loss / train_count:.6f} "
                f"val={val_loss / val_count:.6f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cond_scaler": dataset.cond_scaler.state_dict(),
            "seq_len": args.seq_len,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "cond_keys": COND_KEYS,
        },
        args.output,
    )
    print(f"saved model: {args.output}")


def normalized_to_screen(points, start, target):
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        raise ValueError("start and target must be different")

    theta = math.atan2(float(delta[1]), float(delta[0]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x = points[:, 0] * distance
    y = points[:, 1] * distance
    screen_x = start[0] + x * cos_t - y * sin_t
    screen_y = start[1] + x * sin_t + y * cos_t
    return np.stack([screen_x, screen_y], axis=1)


def generate(args):
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    seq_len = int(checkpoint["seq_len"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrajectoryCVAE(
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cond_scaler = Standardizer.from_state_dict(checkpoint["cond_scaler"])
    condition = np.array([[args.A, args.W, args.ID, args.duration]], dtype=np.float32)
    cond = torch.from_numpy(cond_scaler.transform(condition)).to(device)
    z = torch.randn(args.count, model.latent_dim, device=device) * args.temperature
    cond = cond.expand(args.count, -1)

    with torch.no_grad():
        normalized = model.decode(cond, z, seq_len).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = (args.start_x, args.start_y)
    target = (args.target_x, args.target_y)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        fieldnames = ("sample", "point_index", "x_norm", "y_norm", "x", "y")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, sample in enumerate(normalized):
            screen = normalized_to_screen(sample, start, target)
            for point_index, (norm_xy, xy) in enumerate(zip(sample, screen)):
                writer.writerow(
                    {
                        "sample": sample_index,
                        "point_index": point_index,
                        "x_norm": float(norm_xy[0]),
                        "y_norm": float(norm_xy[1]),
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                    }
                )
    print(f"saved generated trajectories: {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and sample human-like mouse trajectories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--csv", type=Path, default=Path("normalized_output/normalized_trajectory_points.csv"))
    train_parser.add_argument("--output", type=Path, default=Path("models/mouse_trajectory_cvae.pt"))
    train_parser.add_argument("--seq-len", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--beta", type=float, default=0.001)
    train_parser.add_argument("--smooth-weight", type=float, default=0.02)
    train_parser.add_argument("--velocity-weight", type=float, default=0.0)
    train_parser.add_argument("--acceleration-weight", type=float, default=0.0)
    train_parser.add_argument("--acceleration-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--jerk-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--deviation-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--path-length-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--peak-velocity-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--late-acceleration-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--late-jerk-stat-weight", type=float, default=0.0)
    train_parser.add_argument("--late-ratio", type=float, default=0.3)
    train_parser.add_argument("--latent-dim", type=int, default=16)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--val-ratio", type=float, default=0.15)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="")
    train_parser.set_defaults(func=train)

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--model", type=Path, default=Path("models/mouse_trajectory_cvae.pt"))
    gen_parser.add_argument("--output", type=Path, default=Path("generated/trajectories.csv"))
    gen_parser.add_argument("--A", type=float, required=True)
    gen_parser.add_argument("--W", type=float, required=True)
    gen_parser.add_argument("--ID", type=float, default=0.0)
    gen_parser.add_argument("--duration", type=float, default=600.0)
    gen_parser.add_argument("--start-x", type=float, required=True)
    gen_parser.add_argument("--start-y", type=float, required=True)
    gen_parser.add_argument("--target-x", type=float, required=True)
    gen_parser.add_argument("--target-y", type=float, required=True)
    gen_parser.add_argument("--count", type=int, default=5)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--device", default="")
    gen_parser.set_defaults(func=generate)

    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
