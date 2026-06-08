import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BASE_KEYS = ("subject", "subfolder", "condition_index", "trial_number")
OUTPUT_FIELDNAMES = [
    "subject",
    "subfolder",
    "condition_index",
    "trial_number",
    "segment_index",
    "point_index",
    "A",
    "W",
    "ID",
    "duration",
    "error",
    "t_norm",
    "x_norm",
    "y_norm",
]


def _as_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _base_key(row):
    return tuple(row.get(key, "") for key in BASE_KEYS)


def _segment_key(row, segment_index):
    return (*_base_key(row), segment_index)


def collect_segments(input_path):
    last_point_by_group = {}
    segment_by_group = defaultdict(int)
    count_by_segment = defaultdict(int)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group_key = _base_key(row)
            point_index = _as_int(row.get("point_index"))
            last_point = last_point_by_group.get(group_key)

            if last_point is not None and point_index <= last_point:
                segment_by_group[group_key] += 1

            last_point_by_group[group_key] = point_index
            count_by_segment[_segment_key(row, segment_by_group[group_key])] += 1

    return count_by_segment


def write_fixed_csv(input_path, output_path, count_by_segment):
    last_point_by_group = {}
    segment_by_group = defaultdict(int)
    position_by_segment = defaultdict(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for row in reader:
            group_key = _base_key(row)
            point_index = _as_int(row.get("point_index"))
            last_point = last_point_by_group.get(group_key)

            if last_point is not None and point_index <= last_point:
                segment_by_group[group_key] += 1

            segment_index = segment_by_group[group_key]
            segment_key = _segment_key(row, segment_index)
            position = position_by_segment[segment_key]
            count = count_by_segment[segment_key]
            duration = max(count - 1, 1)

            a = _as_float(row.get("A"))
            w = _as_float(row.get("W"))
            fitts_id = math.log2(a / w + 1.0) if w > 0 else 0.0

            writer.writerow(
                {
                    "subject": row.get("subject", ""),
                    "subfolder": row.get("subfolder", ""),
                    "condition_index": row.get("condition_index", ""),
                    "trial_number": row.get("trial_number", ""),
                    "segment_index": segment_index,
                    "point_index": point_index,
                    "A": a,
                    "W": w,
                    "ID": fitts_id,
                    "duration": duration,
                    "error": row.get("error", "False"),
                    "t_norm": position / duration,
                    "x_norm": row.get("x_norm", ""),
                    "y_norm": row.get("y_norm", ""),
                }
            )

            last_point_by_group[group_key] = point_index
            position_by_segment[segment_key] += 1


def repair_csv(input_path, output_path):
    counts = collect_segments(input_path)
    write_fixed_csv(input_path, output_path, counts)
    print(f"segments: {len(counts)}")
    print(f"saved: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Repair normalized trajectory CSV for model training.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("normalized_trajectory_points.csv"))
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("normalized_trajectory_points_fixed.csv"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    repair_csv(args.input, args.output)
