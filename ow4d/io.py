from pathlib import Path
import json
import csv

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)

    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
