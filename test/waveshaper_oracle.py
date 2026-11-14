from typing import Literal

import numpy as np


def waveshaper_audio(
    source: np.ndarray,
    drive: float | np.ndarray = 2.0,
    bias: float | np.ndarray = 0.0,
    output_db: float | np.ndarray = 0.0,
    curve: Literal["soft", "hard", "fold", "half_wave", "full_wave"] = "soft",
) -> np.ndarray:
    """Independent finite convolution oracle, including stage control timing."""
    n = np.arange(65)
    coefficients = 0.5 * np.sinc((n - 32) / 2) * np.blackman(65)
    coefficients /= coefficients.sum()

    def filter_audio(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.convolve(values[:, c], coefficients)[: len(values)]
                for c in range(values.shape[1])
            ]
        )

    def control(value: float | np.ndarray, frames: int, initial: float) -> np.ndarray:
        if np.isscalar(value):
            return np.full((len(source), 1), value)
        values = np.asarray(value).reshape(len(source), 1)
        return np.concatenate((np.full((frames, 1), initial), values[:-frames]))

    values = source
    for _ in range(2):
        up = np.zeros((len(values) * 2, values.shape[1]))
        up[::2] = values
        values = 2 * filter_audio(up)
    drives = np.repeat(control(drive, 24, 2.0), 4, axis=0)
    biases = np.repeat(control(bias, 24, 0.0), 4, axis=0)
    values = drives * values + biases
    if curve == "hard":
        values = np.clip(values, -1, 1) - np.clip(biases, -1, 1)
    elif curve == "soft":
        values = np.tanh(values) - np.tanh(biases)
    elif curve == "half_wave":
        values = (values + np.abs(values) - biases - np.abs(biases)) / 2
    elif curve == "full_wave":
        values = np.sqrt(values * values) - np.sqrt(biases * biases)
    else:
        bands = np.floor((values + 1) / 2)
        bias_bands = np.floor((biases + 1) / 2)
        values = (1 - 2 * (bands % 2)) * (values - 2 * bands)
        values -= (1 - 2 * (bias_bands % 2)) * (biases - 2 * bias_bands)
    for _ in range(2):
        values = filter_audio(values)[::2]
    return values * np.power(10, control(output_db, 48, 0.0) / 20)
