import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_trajectories import iter_generated_metrics, iter_real_metrics, summarize


METRICS_FOR_SCORE = [
    "path_length_ratio_mean",
    "max_abs_deviation_mean",
    "peak_velocity_mean",
    "mean_acceleration_mean",
    "mean_jerk_mean",
]


FILENAME_RE = re.compile(r"cond(?P<cond>\d+)_A(?P<A>\d+)_W(?P<W>\d+)_temp(?P<temp>\d+)p(?P<decimal>\d+)\.csv$")


def relative_gap(generated, real):
    gaps = []
    for metric in METRICS_FOR_SCORE:
        real_value = float(real[metric])
        generated_value = float(generated[metric])
        gaps.append(abs(generated_value - real_value) / (abs(real_value) + 1e-9))
    return float(np.mean(gaps))


def parse_generated_path(path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return {
        "condition_index": int(match.group("cond")),
        "A": float(match.group("A")),
        "W": float(match.group("W")),
        "temperature": float(f"{match.group('temp')}.{match.group('decimal')}"),
    }


def write_rows(path, rows):
    fieldnames = [
        "condition_index",
        "A",
        "W",
        "temperature",
        "real_count",
        "generated_count",
        "avg_relative_gap",
    ]
    metric_names = [
        "path_length_ratio",
        "max_abs_deviation",
        "endpoint_error",
        "peak_velocity",
        "mean_acceleration",
        "mean_jerk",
    ]
    for metric in metric_names:
        fieldnames.extend([f"real_{metric}_mean", f"generated_{metric}_mean"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path, rows):
    conditions = sorted({int(row["condition_index"]) for row in rows})
    temperatures = sorted({float(row["temperature"]) for row in rows})
    matrix = np.full((len(conditions), len(temperatures)), np.nan)

    for row in rows:
        y = conditions.index(int(row["condition_index"]))
        x = temperatures.index(float(row["temperature"]))
        matrix[y, x] = float(row["avg_relative_gap"])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(temperatures)), [f"{temp:.1f}" for temp in temperatures])
    ax.set_yticks(range(len(conditions)), [str(condition) for condition in conditions])
    ax.set_xlabel("temperature")
    ax.set_ylabel("condition index")
    ax.set_title("Original Condition Evaluation: Lower Relative Gap Is Better")
    fig.colorbar(image, ax=ax, label="avg relative gap")
    fig.tight_layout()
    fig.savefig(path)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate generated trajectories across original A/W conditions.")
    parser.add_argument("--real", type=Path, default=Path("normalized_trajectory_points_fixed.csv"))
    parser.add_argument("--generated-dir", type=Path, default=Path("generated/original_conditions_seq128_dyn"))
    parser.add_argument("--summary", type=Path, default=Path("generated/original_conditions_seq128_dyn/evaluation_by_condition.csv"))
    parser.add_argument("--heatmap", type=Path, default=Path("generated/original_conditions_seq128_dyn/evaluation_heatmap.png"))
    return parser


def main():
    args = build_parser().parse_args()
    real_cache = {}
    rows = []

    for path in sorted(args.generated_dir.glob("*.csv")):
        meta = parse_generated_path(path)
        if meta is None:
            continue

        condition_key = (meta["A"], meta["W"])
        if condition_key not in real_cache:
            real_metrics = list(iter_real_metrics(args.real, filter_a=meta["A"], filter_w=meta["W"]))
            real_cache[condition_key] = summarize(real_metrics)

        real_summary = real_cache[condition_key]
        generated_summary = summarize(list(iter_generated_metrics(path)))
        row = {
            "condition_index": meta["condition_index"],
            "A": meta["A"],
            "W": meta["W"],
            "temperature": meta["temperature"],
            "real_count": real_summary["count"],
            "generated_count": generated_summary["count"],
            "avg_relative_gap": relative_gap(generated_summary, real_summary),
        }

        for metric in [
            "path_length_ratio",
            "max_abs_deviation",
            "endpoint_error",
            "peak_velocity",
            "mean_acceleration",
            "mean_jerk",
        ]:
            row[f"real_{metric}_mean"] = real_summary[f"{metric}_mean"]
            row[f"generated_{metric}_mean"] = generated_summary[f"{metric}_mean"]

        rows.append(row)

    write_rows(args.summary, rows)
    plot_heatmap(args.heatmap, rows)

    by_temperature = {}
    for row in rows:
        by_temperature.setdefault(row["temperature"], []).append(row["avg_relative_gap"])

    print(f"saved summary: {args.summary}")
    print(f"saved heatmap: {args.heatmap}")
    print("temperature ranking:")
    for temperature, gaps in sorted(by_temperature.items(), key=lambda item: float(np.mean(item[1]))):
        print(f"temp={temperature:.1f} mean_gap={float(np.mean(gaps)):.3f}")

    print("best per condition:")
    for condition in sorted({row["condition_index"] for row in rows}):
        condition_rows = [row for row in rows if row["condition_index"] == condition]
        best = min(condition_rows, key=lambda row: row["avg_relative_gap"])
        print(
            f"cond={condition:02d} A={best['A']:g} W={best['W']:g} "
            f"best_temp={best['temperature']:.1f} gap={best['avg_relative_gap']:.3f}"
        )


if __name__ == "__main__":
    main()
