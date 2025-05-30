"""entry point"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_raman import load_spectrum
from interpolate_raman import interpolate_to_grid
from normalize_raman import minmax_scale


def run_pipeline(
    file_path: Path,
    plot: bool = False,
    save_npy: bool = False,
    out_stem: str | None = None,
) -> None:

    raw_df = load_spectrum(file_path)
    print(f"Loaded {len(raw_df)} raw points from {file_path.name}")

    interp_df, y_interp = interpolate_to_grid(raw_df)
    print("Interpolated to 571 points (50–1750 cm⁻¹)")

    y_norm = minmax_scale(y_interp)
    interp_df["intensity_norm"] = y_norm
    print("Per-spectrum min-max normalisation complete")

    if plot:
        interp_df.plot(
            x="wavenumber_cm⁻¹",
            y="intensity_norm",
            figsize=(6, 4),
        )
        plt.gca()
        plt.title(f"Normalised spectrum: {file_path.name}")
        plt.tight_layout()
        plt.show()


    if save_npy:
        stem = out_stem or f"{file_path.stem}_norm"
        npy_path = file_path.with_name(f"{stem}.npy")
        np.save(npy_path, y_norm.astype(np.float32))
        print(f"Saved 571-point normalised vector → {npy_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Raman spectrum, interpolate to a 571-point grid, "
            "min-max normalise, and optionally plot or save the result."
        )
    )
    parser.add_argument("spectrum_file", type=Path, help="Path to raw .txt or .csv file")
    parser.add_argument("--plot", action="store_true", help="Show a quick plot")
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save the 571-point normalised vector as <stem>_norm.npy",
    )
    parser.add_argument(
        "--out",
        dest="out_stem",
        help="Custom filename stem for the saved .npy (default: <input>_norm)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        file_path=args.spectrum_file,
        plot=args.plot,
        save_npy=args.save_npy,
        out_stem=args.out_stem,
    )
