"""
robust loader  ▸  interpolate to 571 points  ▸  min-max normalise (0-1)

Works for:
• RRUFF .txt  – skips metadata lines
• MLROD .csv – auto-detects shift & intensity columns
Compatible with NumPy ≥ 2.0 (uses np.ptp)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

TARGET_RANGE = (50, 1750)
N_SAMPLES = 571  # Qi et al. use 571 points

# ---------------------------------------------------------------------------
# 1 · generic spectrum reader
# ---------------------------------------------------------------------------
def _read_generic(fp: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['shift', 'intensity'].
    Auto-handles RRUFF .txt and MLROD .csv variations.
    """
    if fp.suffix == ".txt":                                    # ----- RRUFF
        rows = []
        with open(fp) as fh:
            for line in fh:
                parts = re.split(r"[,\s]+", line.strip())
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue                                   # skip headers
        df = pd.DataFrame(rows, columns=["shift", "intensity"])

    else:                                                      # ----- MLROD
        try:
            df = pd.read_csv(fp, engine="python")              # tolerate ragged rows
        except pd.errors.ParserError:
            # fallback: whitespace-delimited (rare)
            df = pd.read_table(fp, engine="python")

        # find likely column names
        shift_col = next(c for c in df.columns if "shift" in c.lower())
        inten_col = next(c for c in df.columns
                         if "inten" in c.lower() or "counts" in c.lower())
        df = df[[shift_col, inten_col]].rename(
            columns={shift_col: "shift", inten_col: "intensity"}
        )

    # ---- enforce strictly increasing x ------------------------------------
    df = (
        df.dropna()
          .sort_values("shift", kind="mergesort")
          .drop_duplicates(subset="shift", keep="first")
          .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# 2 · interpolate + normalise
# ---------------------------------------------------------------------------
def interp_and_norm(fp: Path) -> np.ndarray:
    df = _read_generic(fp)
    if len(df) < 4:
        raise ValueError(f"{fp.name}: too few data points after cleaning")

    grid = np.linspace(*TARGET_RANGE, N_SAMPLES)
    y_u = UnivariateSpline(df["shift"], df["intensity"], s=0)(grid)

    # min-max → 0-1  (NumPy ≥ 2.0 safe)
    y_u = (y_u - y_u.min()) / np.ptp(y_u)
    return y_u.astype("float32")
