"""helper to resample raman spectrum onto the 50–1750 cm⁻¹ grid
(571 points) used in the reference paper"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate_to_grid(
    raman_data_df: pd.DataFrame,
    target_min: float = 50.0,
    target_max: float = 1750.0,
    n_points: int = 571,
    kind: str = "cubic",
    fill_value: float | tuple | str = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    
    x_old = raman_data_df["wavenumber_cm⁻¹"].to_numpy()
    y_old = raman_data_df["intensity"].to_numpy()


    if np.any(np.diff(x_old) <= 0):
        raise ValueError(
            "wavenumber_cm⁻¹ must be strictly increasing; check input order."
        )

    x_new = np.linspace(target_min, target_max, n_points, dtype=np.float64)
    f = interp1d(
        x_old,
        y_old,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value,
    )
    y_new = f(x_new)

    interp_df = pd.DataFrame(
        {"wavenumber_cm⁻¹": x_new, "intensity": y_new}
    )
    return interp_df, y_new
