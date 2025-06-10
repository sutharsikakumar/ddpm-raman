"""helper to load raman text file"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_spectrum(
    path: str | Path,
    header_marker: str = "#",
    col_names: tuple[str, str] = ("wavenumber_cm⁻¹", "intensity"),
    plot: bool = False,
) -> pd.DataFrame:
    
    df = pd.read_csv(
        path,
        delim_whitespace=True, 
        comment=header_marker,
        names=list(col_names),
        skip_blank_lines=True,
        engine="python",
    ).sort_values(col_names[0], ascending=True).reset_index(drop=True)

    if plot:
        df.plot(x=col_names[0], y=col_names[1], figsize=(6, 4))
        plt.tight_layout()
        plt.show()

    return df

# test
if __name__ == "__main__":
    TEST_FILE = "aa-raman-551.txt"
    df = load_spectrum(TEST_FILE, plot=True)
    print(df.head())
