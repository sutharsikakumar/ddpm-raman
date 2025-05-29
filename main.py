"""entry point"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from load_raman import load_spectrum
from interpolate_raman import interpolate_to_grid


def run_pipeline(
    file_path: Path,
    plot: bool = False,
    save_npy: bool = False,
    out_stem: str | None = None,
) -> None:
  
    raw_df = load_spectrum(file_path)
    print(f"Loaded {len(raw_df)} raw points from {file_path.name}")

    interp_df, y_new = interpolate_to_grid(raw_df)
    print("Interpolated to 571 points (50–1750 cm⁻¹)")

    if plot:
        interp_df.plot(x="wavenumber_cm⁻¹", y="intensity", figsize=(6, 4))
        plt.title(f"Interpolated spectrum: {file_path.name}")
        plt.tight_layout()
        plt.show()

    if save_npy:
        stem = out_stem or file_path.stem + "_interp"
        npy_path = file_path.with_name(f"{stem}.npy")
        np.save(npy_path, y_new.astype(np.float32))
        print(f"Saved 571-point vector ➜ {npy_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load Raman spectrum, interpolate to 571-point grid."
    )
    p.add_argument("spectrum_file", type=Path, help="Path to raw .txt or .csv")
    p.add_argument("--plot", action="store_true", help="Show a quick plot")
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="Save the 571-point vector as <stem>.npy",
    )
    p.add_argument(
        "--out",
        dest="out_stem",
        help="Custom filename stem for saved .npy (default: <input>_interp)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        file_path=args.spectrum_file,
        plot=args.plot,
        save_npy=args.save_npy,
        out_stem=args.out_stem,
    )