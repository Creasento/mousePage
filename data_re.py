from pathlib import Path
import csv
import re
import math
import xml.etree.ElementTree as ET

def parse_point(text):
    # "{X=1280, Y=570, Radius=10}" or "{X=1282, Y=570, Time=0}"
    pairs = re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", text)
    return {k: float(v) for k, v in pairs}


def normalize_xy(x, y, last_x, last_y, this_x, this_y):
    dx = this_x - last_x
    dy = this_y - last_y
    A = math.sqrt(dx * dx + dy * dy)

    theta = math.atan2(dy, dx)

    px = x - last_x
    py = y - last_y

    x_rot = px * math.cos(theta) + py * math.sin(theta)
    y_rot = -px * math.sin(theta) + py * math.cos(theta)

    return x_rot / A, y_rot / A


def process_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    subject = root.attrib.get("subject", None)

    results = []

    for cond in root.findall("Condition"):
        condition_data = {
            "condition_block": int(cond.attrib.get("block", -1)),
            "condition_index": int(cond.attrib.get("index", -1)),
            "condition_A": float(cond.attrib.get("A", -1)),
            "condition_W": float(cond.attrib.get("W", -1)),
            "condition_ID": float(cond.attrib.get("ID", -1)),
        }

        for trial in cond.findall("Trial"):
            last = parse_point(trial.attrib["lastCircle"])
            this = parse_point(trial.attrib["thisCircle"])

            last_x, last_y = last["X"], last["Y"]
            this_x, this_y = this["X"], this["Y"]

            dx = this_x - last_x
            dy = this_y - last_y
            A_real = math.sqrt(dx * dx + dy * dy)
            theta = math.atan2(dy, dx)

            movement = trial.find("Movement")
            if movement is None:
                continue

            traj = []

            for move in movement.findall("move"):
                p = parse_point(move.attrib["point"])

                x_norm, y_norm = normalize_xy(
                    p["X"], p["Y"],
                    last_x, last_y,
                    this_x, this_y
                )

                traj.append({
                    "index": int(move.attrib["index"]),
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "time": p["Time"],
                })

            if len(traj) == 0:
                continue

            t0 = traj[0]["time"]
            t1 = traj[-1]["time"]
            duration = max(t1 - t0, 1)

            for point in traj:
                point["t_norm"] = (point["time"] - t0) / duration

            item = {
                "source_file": str(xml_path),
                "subject": subject,
                "participant_folder": xml_path.parents[1].name,  # P1, P2 ...
                "subfolder": xml_path.parents[0].name,            # 2,3,4...
                **condition_data,

                "trial_number": int(trial.attrib.get("number", -1)),
                "A": float(trial.attrib.get("A", A_real)),
                "W": float(trial.attrib.get("W", condition_data["condition_W"])),
                "ID": condition_data["condition_ID"],
                "angle": float(trial.attrib.get("angle", math.degrees(theta))),
                "axis": float(trial.attrib.get("axis", -1)),

                "lastCircle": last,
                "thisCircle": this,
                "A_real": A_real,
                "theta_rad": theta,
                "radius_norm": this.get("Radius", 0) / A_real,

                "MTe": float(trial.attrib.get("MTe", -1)),
                "error": trial.attrib.get("error", "false") == "true",
                "overshoots": int(trial.attrib.get("overshoots", -1)),
                "entries": int(trial.attrib.get("entries", -1)),

                "movement_count": int(movement.attrib.get("count", len(traj))),
                "travel": float(movement.attrib.get("travel", -1)),
                "duration": float(movement.attrib.get("duration", -1)),
                "submovements": int(movement.attrib.get("submovements", -1)),

                "trajectory": traj,
            }

            results.append(item)

    return results


CSV_FIELDNAMES = [
    "subject",
    "subfolder",
    "condition_index",
    "trial_number",
    "point_index",
    "A",
    "W",
    "error",
    "x_norm",
    "y_norm",
]


def iter_csv_rows(trial):
    trial_row = {key: trial[key] for key in CSV_FIELDNAMES if key in trial}

    for point in trial["trajectory"]:
        yield {
            **trial_row,
            "point_index": point["index"],
            "x_norm": point["x_norm"],
            "y_norm": point["y_norm"],
        }


def normalize_all(root_dir, output_path):
    root_dir = Path(root_dir)
    output_path = Path(output_path)

    xml_files = sorted(root_dir.glob("P*/[2-8]/s01_2D_nomet__*.xml"))

    print(f"Found XML files: {len(xml_files)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    trial_count = 0
    row_count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for xml_path in xml_files:
            try:
                trials = process_xml(xml_path)

                for trial in trials:
                    for row in iter_csv_rows(trial):
                        writer.writerow(row)
                        row_count += 1

                trial_count += len(trials)
                print(f"[OK] {xml_path} -> {len(trials)} trials")
            except Exception as e:
                print(f"[ERROR] {xml_path}: {e}")

    print(f"\nSaved: {output_path}")
    print(f"Total trials: {trial_count}")
    print(f"Total trajectory rows: {row_count}")


if __name__ == "__main__":
    normalize_all(
        root_dir=Path(__file__).with_name("fittsdata"),
        output_path="normalized_output/normalized_trajectory_points.csv"
    )
