import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REAL_GROUP_KEYS = ("subject", "subfolder", "condition_index", "trial_number", "segment_index")
METRIC_NAMES = [
    "path_length_ratio",
    "max_abs_deviation",
    "mean_abs_deviation",
    "endpoint_error",
    "overshoot_x",
    "undershoot_x",
    "peak_velocity",
    "time_to_peak_velocity",
    "mean_velocity",
    "mean_acceleration",
    "mean_jerk",
    "late_speed_ratio",
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trajectory_metrics(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return None

    diffs = np.diff(points, axis=0)
    step_dist = np.linalg.norm(diffs, axis=1)
    path_length = float(step_dist.sum())

    velocity = step_dist
    acceleration = np.diff(diffs, axis=0)
    accel_mag = np.linalg.norm(acceleration, axis=1)
    jerk = np.diff(acceleration, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    late_start = max(0, int(len(velocity) * 0.8))
    mean_velocity = float(velocity.mean()) if len(velocity) else 0.0
    late_velocity = float(velocity[late_start:].mean()) if len(velocity[late_start:]) else 0.0

    end = points[-1]
    peak_index = int(np.argmax(velocity)) if len(velocity) else 0
    denom = max(len(velocity) - 1, 1)

    return {
        "path_length_ratio": path_length,
        "max_abs_deviation": float(np.max(np.abs(points[:, 1]))),
        "mean_abs_deviation": float(np.mean(np.abs(points[:, 1]))),
        "endpoint_error": float(np.linalg.norm(end - np.array([1.0, 0.0]))),
        "overshoot_x": float(max(0.0, np.max(points[:, 0]) - 1.0)),
        "undershoot_x": float(max(0.0, -np.min(points[:, 0]))),
        "peak_velocity": float(np.max(velocity)) if len(velocity) else 0.0,
        "time_to_peak_velocity": peak_index / denom,
        "mean_velocity": mean_velocity,
        "mean_acceleration": float(accel_mag.mean()) if len(accel_mag) else 0.0,
        "mean_jerk": float(jerk_mag.mean()) if len(jerk_mag) else 0.0,
        "late_speed_ratio": late_velocity / mean_velocity if mean_velocity > 0 else 0.0,
    }


def iter_real_metrics(path, filter_a=None, filter_w=None, tolerance=1e-6):
    groups = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error", "false").lower() == "true":
                continue
            if filter_a is not None and abs(_safe_float(row.get("A")) - filter_a) > tolerance:
                continue
            if filter_w is not None and abs(_safe_float(row.get("W")) - filter_w) > tolerance:
                continue
            key = tuple(row.get(k, "") for k in REAL_GROUP_KEYS)
            groups[key].append(
                (
                    _safe_float(row.get("point_index")),
                    _safe_float(row.get("x_norm")),
                    _safe_float(row.get("y_norm")),
                )
            )

    for points in groups.values():
        points.sort(key=lambda item: item[0])
        metric = trajectory_metrics([(x, y) for _, x, y in points])
        if metric is not None:
            yield metric


def iter_generated_metrics(path):
    groups = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups[row.get("sample", "")].append(
                (
                    _safe_float(row.get("point_index")),
                    _safe_float(row.get("x_norm")),
                    _safe_float(row.get("y_norm")),
                )
            )

    for points in groups.values():
        points.sort(key=lambda item: item[0])
        metric = trajectory_metrics([(x, y) for _, x, y in points])
        if metric is not None:
            yield metric


def summarize(metrics):
    values = {name: np.array([m[name] for m in metrics], dtype=np.float64) for name in METRIC_NAMES}
    summary = {"count": len(metrics)}
    for name, arr in values.items():
        summary[f"{name}_mean"] = float(arr.mean()) if len(arr) else float("nan")
        summary[f"{name}_std"] = float(arr.std()) if len(arr) else float("nan")
        summary[f"{name}_p10"] = float(np.percentile(arr, 10)) if len(arr) else float("nan")
        summary[f"{name}_p50"] = float(np.percentile(arr, 50)) if len(arr) else float("nan")
        summary[f"{name}_p90"] = float(np.percentile(arr, 90)) if len(arr) else float("nan")
    return summary


def write_summary(path, rows):
    fieldnames = ["dataset", "count"]
    for name in METRIC_NAMES:
        fieldnames.extend(
            [
                f"{name}_mean",
                f"{name}_std",
                f"{name}_p10",
                f"{name}_p50",
                f"{name}_p90",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_comparison(path, all_metrics):
    metrics_to_plot = [
        "path_length_ratio",
        "max_abs_deviation",
        "endpoint_error",
        "peak_velocity",
        "mean_acceleration",
        "mean_jerk",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    real = all_metrics["real"]

    for ax, metric_name in zip(axes.flat, metrics_to_plot):
        real_values = np.array([m[metric_name] for m in real], dtype=np.float64)
        lo, hi = np.percentile(real_values, [1, 99])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
            lo, hi = float(real_values.min()), float(real_values.max())
        bins = np.linspace(lo, hi, 35)
        ax.hist(real_values, bins=bins, density=True, alpha=0.35, color="black", label="real")

        for dataset, metrics in all_metrics.items():
            if dataset == "real":
                continue
            values = np.array([m[metric_name] for m in metrics], dtype=np.float64)
            ax.axvline(values.mean(), linewidth=1.2, alpha=0.75, label=dataset)

        ax.set_title(metric_name)
        ax.grid(True, alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate generated mouse trajectories against real trajectories.")
    parser.add_argument("--real", type=Path, default=Path("normalized_trajectory_points_fixed.csv"))
    parser.add_argument("--generated-dir", type=Path, default=Path("generated"))
    parser.add_argument("--summary", type=Path, default=Path("generated/evaluation_summary.csv"))
    parser.add_argument("--plot", type=Path, default=Path("generated/evaluation_metric_comparison.png"))
    parser.add_argument("--real-A", type=float, default=None)
    parser.add_argument("--real-W", type=float, default=None)
    parser.add_argument("--filter-tolerance", type=float, default=1e-6)
    return parser


def main():
    args = build_parser().parse_args()
    all_metrics = {
        "real": list(
            iter_real_metrics(
                args.real,
                filter_a=args.real_A,
                filter_w=args.real_W,
                tolerance=args.filter_tolerance,
            )
        )
    }

    for generated_path in sorted(args.generated_dir.glob("temperature_*.csv")):
        temp_label = generated_path.stem.removeprefix("temperature_").replace("p", ".")
        dataset = f"temp={temp_label}"
        all_metrics[dataset] = list(iter_generated_metrics(generated_path))

    rows = []
    for dataset, metrics in all_metrics.items():
        row = {"dataset": dataset}
        row.update(summarize(metrics))
        rows.append(row)

    write_summary(args.summary, rows)
    plot_metric_comparison(args.plot, all_metrics)

    print(f"saved summary: {args.summary}")
    print(f"saved plot: {args.plot}")
    for row in rows:
        print(
            f"{row['dataset']:>8} n={row['count']} "
            f"path={row['path_length_ratio_mean']:.3f} "
            f"dev={row['max_abs_deviation_mean']:.3f} "
            f"end={row['endpoint_error_mean']:.3f} "
            f"jerk={row['mean_jerk_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
