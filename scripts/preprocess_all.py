#!/usr/bin/env python3
from pathlib import Path
import numpy as np, traceback
from raman_ddpm.interp_norm import interp_and_norm

RAW_FOLDERS = [
    Path("data/rruff/raw_txt"),
    Path("data/mlrod/raw_csv"),
]
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

count_ok, count_bad = 0, 0
bad_list = []

for folder in RAW_FOLDERS:
    if not folder.exists():
        continue
    for fp in folder.glob("*.*"):
        try:
            arr = interp_and_norm(fp)
            np.save(OUT_DIR / f"{fp.stem}.npy", arr)
            count_ok += 1
        except Exception as e:
            count_bad += 1
            bad_list.append(f"{fp.name} → {e}")

            continue

print(f"Saved {count_ok} spectra to {OUT_DIR}")
print(f"Skipped {count_bad} malformed files (see bad_files.log)")


with open("bad_files.log", "w") as fh:
    fh.write("\n".join(bad_list))
