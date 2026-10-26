"""Extract recorded ETTh1 supervised training and validation MSE loss by epoch.

Run from the repository root with: python scripts/extract_etth1_epochs.py
The final test MSE/MAE is stored separately in results/supervised.csv.
"""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
OUTPUT = ROOT / "results" / "etth1_epoch_losses.csv"
MAIN_OUTPUT = ROOT / "results" / "etth1_patchtst42_96_epoch_losses.csv"
EPOCH_LINE = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*"
    r"Train Loss:\s*([\d.eE+-]+)\s+Vali Loss:\s*([\d.eE+-]+)"
)


def main():
    rows = []
    matched_logs = 0
    for path in sorted(LOG_DIR.glob("*__ETTh1__*.log")):
        # The masked pre-training/probe logs have a different training loop.
        if path.name.startswith("F__"):
            continue
        run_tag = path.stem.replace("__", "/")
        log_rows = [
            (run_tag, int(epoch), train, validation)
            for epoch, train, validation in EPOCH_LINE.findall(path.read_text(errors="replace"))
        ]
        if not log_rows:
            raise ValueError(f"No supervised epoch losses found in {path}")
        matched_logs += 1
        rows.extend(log_rows)

    if not rows:
        raise ValueError(f"No ETTh1 supervised epoch logs found in {LOG_DIR}")

    with OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["run_tag", "epoch", "train_mse_loss", "validation_mse_loss"])
        writer.writerows(rows)

    main_rows = [row[1:] for row in rows if row[0] == "A/PatchTST42/ETTh1/96"]
    if not main_rows:
        raise ValueError("Missing the main ETTh1/96 PatchTST run")
    with MAIN_OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["epoch", "train_mse_loss", "validation_mse_loss"])
        writer.writerows(main_rows)
    print(f"Wrote {len(rows)} epochs from {matched_logs} runs to {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {len(main_rows)} epochs for the main ETTh1/96 plot to {MAIN_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
