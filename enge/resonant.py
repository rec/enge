"""Fixed Karplus-style tuning and nominal undamped decay coefficients."""

from math import atan2, exp, floor, isfinite, log, pi, tan

from . import highpass


def tuning(frequency: float, damping: float, sample_rate: int) -> tuple[int, float]:
    if not isfinite(frequency) or not 0 < frequency <= sample_rate / 4:
        raise ValueError("Resonant frequency must be positive and at most quarter-rate")
    highpass.coefficient(damping, sample_rate)
    omega = 2 * pi * frequency / sample_rate
    phase = atan2(tan(omega / 2), tan(pi * damping / sample_rate))
    period = (2 * pi - phase) / omega
    if not isfinite(period) or period >= 2**63:
        raise ValueError("Unrepresentable resonant delay capacity")
    frames = floor(period - 0.5)
    residual = (2 * pi - phase) - frames * omega
    q = tan(residual / 2) / tan(omega / 2)
    a = (1 - q) / (1 + q)
    if frames < 1 or not 0 < residual < pi or not -1 < a < 1:
        raise ValueError("Unrepresentable resonant tuning coefficient")
    return frames, a


def loss(frequency: float, decay: float) -> float:
    if not isfinite(decay) or decay <= 0:
        raise ValueError("Resonant decay must be finite and positive")
    gain = exp((-3 * log(10) / frequency) / decay)
    if not 0 <= gain < 1:
        raise ValueError("Unrepresentable resonant decay")
    return gain
