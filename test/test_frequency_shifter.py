from math import fsum
from pathlib import Path

import numpy as np
import pytest
from test_effect_envelfo import modulated_graph
from test_modulated_multitap import replay_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def graph(shift: float = 250) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main", channels=["left", "right"], source=audio_effects.MainStream()
        ),
        [audio_effects.FrequencyShifter(name="shift", shift_hz=shift)],
        48000,
    )


def translated_audio(
    source: np.ndarray, shifts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Direct whole-signal convolution is independent of the streaming paired-tap FIR.
    positions = np.arange(1025) - 512
    taps = np.zeros(1025)
    taps[1::2] = 2 / (np.pi * positions[1::2])
    taps *= np.blackman(1025)
    quadrature = np.column_stack(
        [np.convolve(source[:, c], taps)[:48000] for c in range(2)]
    )
    real = np.vstack([np.zeros((512, 2)), source])[:48000]
    # Sum short blocks and their totals accurately instead of accumulating an
    # unbounded floating phase, which drifts beyond the audio comparison tolerance.
    phase = np.empty(48000)
    totals: list[float] = []
    for start in range(0, 48000, 128):
        block = shifts[start : start + 128]
        phase[start : start + len(block)] = (
            fsum(totals) + np.concatenate([[0], np.cumsum(block[:-1])])
        ) / 48000
        totals.append(fsum(block))
    angle = 2 * np.pi * phase[:, None]
    return real, real * np.cos(angle) - quadrature * np.sin(angle)


def source_audio() -> np.ndarray:
    frames = np.arange(48000)
    source = np.column_stack(
        [
            0.1 * np.cos(2 * np.pi * 1000 * frames / 48000),
            0.07 * np.sin(2 * np.pi * 3000 * frames / 48000),
        ]
    )
    source[36003:] = 0
    return source


@pytest.mark.parametrize("shift", [-2000, -250, 0, 250, 2000])
def test_frequency_translation_matches_fir_and_continuous_phase_across_replay(
    tmp_path: Path, shift: float
) -> None:
    definition = graph(shift)
    assert effects.prepare(definition, 48000).latency_frames == 512
    source = source_audio()
    real, expected = translated_audio(source, np.full(48000, shift))
    if shift == 0:
        np.testing.assert_array_equal(expected, real)
    replay_audio(
        tmp_path,
        definition,
        source,
        [audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")],
        expected,
        True,
    )


def test_frequency_shift_tracks_scoped_motion_ramp_and_warm_bypass(
    tmp_path: Path,
) -> None:
    raw = graph().model_dump()
    modulation = modulated_graph("gain")[0].model_dump()
    raw.update({n: modulation[n] for n in ("envelfos", "bindings", "modulation")})
    target = {"name": "shift", "parameter": "shift_hz"}
    raw["modulation"]["parameters"][0].update(
        target=target, unit="hz_offset", minimum=-2000, maximum=2000, default=250
    )
    raw["modulation"]["routes"][0].update(
        target=target,
        unit="hz_offset",
        points=[{"input": -1, "amount": -50}, {"input": 1, "amount": 50}],
    )
    definition = audio_effects.EffectGraph.model_validate(raw)
    frames = np.arange(48000)
    shifts = (
        250
        - 500 * np.clip((frames - 6001) / 18000, 0, 1)
        + 50 * np.where((frames // 12000) % 2 == 0, 1, -1)
    )
    source = source_audio()
    dry, wet = translated_audio(source, shifts)
    # Mix/bypass commands follow audio through the declared intrinsic latency.
    mix = (
        1
        - np.clip((frames - 15003 - 512) / 64, 0, 1)
        + np.clip((frames - 27007 - 512) / 64, 0, 1)
    )
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=6001,
            ordinal=0,
            processor="shift",
            parameter="shift_hz",
            value=-250,
            duration_frames=18000,
        ),
        audio_effects.BypassAction(
            tick=15003, ordinal=0, processor="shift", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=27007, ordinal=0, processor="shift", bypassed=False
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    replay_audio(
        tmp_path, definition, source, actions, dry + mix[:, None] * (wet - dry), True
    )


def test_frequency_shift_reflects_at_dc_aliases_at_nyquist_and_stops(
    tmp_path: Path,
) -> None:
    frames = np.arange(48000)
    source = np.column_stack(
        [0.1 * np.cos(2 * np.pi * f * frames / 48000) for f in [500, 23500]]
    )
    _, expected = translated_audio(source, np.full(48000, 2000))
    expected[36003:] = 0
    replay_audio(
        tmp_path,
        graph(2000),
        source,
        [audio_effects.StopAction(tick=36003, ordinal=0)],
        expected,
        True,
    )
    # The 25.5 kHz translated right channel folds to 22.5 kHz at 48 kHz.
    window = expected[2400:26400]
    spectrum = np.abs(np.fft.rfft(window, axis=0))
    assert np.argmax(spectrum[:, 0]) * 2 == 2500
    assert np.argmax(spectrum[:, 1]) * 2 == 22500


def test_hilbert_profile_rejects_unwanted_sideband_in_declared_band(
    tmp_path: Path,
) -> None:
    source = source_audio()
    _, expected = translated_audio(source, np.full(48000, 250))
    replay_audio(
        tmp_path,
        graph(),
        source,
        [audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")],
        expected,
        True,
    )
    spectrum = np.abs(np.fft.rfft(expected[2400:26400], axis=0))
    assert spectrum[750 // 2, 0] / spectrum[1250 // 2, 0] < 0.001
    # Check the specified FIR response across the usable band, including both edges.
    taps = np.zeros(1025)
    taps[1::2] = 2 / (np.pi * (np.arange(1, 1025, 2) - 512))
    taps *= np.blackman(1025)
    frequencies = np.linspace(200, 23800, 2001)
    response = (
        np.exp(-2j * np.pi * frequencies[:, None] * np.arange(1025) / 48000) @ taps
    )
    magnitude = np.abs(response)
    assert np.max(np.abs(1 - magnitude) / (1 + magnitude)) < 0.001


def test_frequency_shifter_aligns_internal_mix_and_parallel_dry_branch(
    tmp_path: Path,
) -> None:
    raw = graph().model_dump()
    raw["processors"][0]["mix"] = 0.5
    raw["processors"].append(
        audio_effects.Crossfader(name="blend", law="linear", position=0.5).model_dump()
    )
    raw["connections"].extend(
        [
            {
                "processor": "blend",
                "port": "first",
                "source": {"kind": "processor", "processor": "shift"},
            },
            {
                "processor": "blend",
                "port": "second",
                "source": {"kind": "input", "input": "main"},
            },
        ]
    )
    raw["output"] = {"kind": "processor", "processor": "blend"}
    source = source_audio()
    dry, wet = translated_audio(source, np.full(48000, 250))
    replay_audio(
        tmp_path,
        audio_effects.EffectGraph.model_validate(raw),
        source,
        [audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")],
        dry * 0.75 + wet * 0.25,
        True,
    )


def test_failed_frequency_shift_callback_silences_latches_and_restores(
    tmp_path: Path,
) -> None:
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(effects.prepare(graph(), 48000)),
        2,
        input_routes=[[-2]],
    )
    saved = runtime.snapshot()
    runtime.submit_effects(np.array([[12003, 0, 0, 0, 2001, 0]], dtype=np.float64))
    source = source_audio()
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="frequency shift"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    _, expected = translated_audio(source, np.full(48000, 250))
    check_audio(tmp_path / "restored.wav", actual, expected)
