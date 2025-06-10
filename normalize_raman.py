"""per-spectrum min-max scaling for Raman intensities"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def minmax_scale(
    intensities: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-12,
) -> np.ndarray:

    y = intensities.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(y).ravel()

    if np.isnan(scaled).any():
        scaled = np.zeros_like(intensities, dtype=np.float32)
    return scaled.astype(np.float32)
