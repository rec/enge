import numpy as np

def render(
    waveform: int,
    duty: float,
    rate: float,
    prepared_gain: float,
    frequencies: np.ndarray,
    states: np.ndarray,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]: ...
