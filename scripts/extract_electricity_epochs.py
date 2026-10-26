"""Extract Electricity training and validation MSE losses from supervised logs.

Run from the repository root with: python scripts/extract_electricity_epochs.py
These are split losses, not the final test MSE/MAE in results/supervised.csv.
"""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
OUTPUT = ROOT / "results" / "electricity_epoch_losses.csv"
EPOCH_LINE = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*"
    r"Train Loss:\s*([\d.eE+-]+)\s+Vali Loss:\s*([\d.eE+-]+)"
)
MAIN_RUNS = {
    "X/PatchTST42-full/electricity/96": "electricity_patchtst42_full_96_epoch_losses.csv",
    "E/PatchTST42-32ch/electricity/96": "electricity_patchtst42_32ch_96_epoch_losses.csv",
}


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    rows = []
    for path in sorted(LOG_DIR.glob("*__electricity__*.log")):
        run_tag = path.stem.replace("__", "/")
        found = EPOCH_LINE.findall(path.read_text(errors="replace"))
        if not found:
            raise ValueError(f"No epoch losses found in {path}")
        rows.extend((run_tag, int(epoch), train, validation)
                    for epoch, train, validation in found)
    if not rows:
        raise ValueError(f"No Electricity logs found in {LOG_DIR}")

    write_csv(OUTPUT, ["run_tag", "epoch", "train_mse_loss", "validation_mse_loss"], rows)
    print(f"Wrote {len(rows)} epochs to {OUTPUT.relative_to(ROOT)}")
    for run_tag, filename in MAIN_RUNS.items():
        selected = [row[1:] for row in rows if row[0] == run_tag]
        if not selected:
            raise ValueError(f"Missing {run_tag}")
        path = ROOT / "results" / filename
        write_csv(path, ["epoch", "train_mse_loss", "validation_mse_loss"], selected)
        print(f"Wrote {len(selected)} epochs to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
