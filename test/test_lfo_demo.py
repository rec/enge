from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import change, onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from test_synth_demo import publish_flac
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import instrument, trace
from ufor.synth import SynthInstrumentScore

from enge import sample_instrument, synth


def test_lfo_demo_vibrato_tremolo_and_live_gain(
    tmp_path: Path, pytestconfig: pytest.Config, backend: Literal["numpy", "native"]
) -> None:
    """Two seconds: sine synth on the left, sampled harmonic tone on the right."""
    length = 96000
    frames = np.arange(length)
    value = np.cos(2 * np.pi * frames / 12000)
    weight = np.clip((frames - 6000) / 6000, 0, 1)
    ratios = np.exp2(60 * weight * value / 1200)
    gain = (1 + weight * (-0.375 + 0.375 * value)) * (
        1 - 0.4 * np.clip((frames - 48000) / 2400, 0, 1)
    )
    envelope = (
        0.3 * np.minimum(frames / 960, 1) * np.clip(1 - (frames - 84000) / 12000, 0, 1)
    )
    # Higher-precision integration is independent of the engines' compensated
    # per-frame recurrence. The sampler oracle linearly interpolates this source.
    positions = np.concatenate(
        [np.zeros(1), np.cumsum(ratios[:-1], dtype=np.longdouble)]
    ).astype(np.float64)
    source_frames = np.arange(120000)
    source = np.sin(2 * np.pi * source_frames * 220 / 48000) + 0.3 * np.sin(
        2 * np.pi * source_frames * 440 / 48000
    )
    expected = (
        np.column_stack(
            [
                np.sin(2 * np.pi * positions * 220 / 48000),
                np.interp(positions, source_frames, source),
            ]
        )
        * (gain * envelope)[:, None]
    )
    actual = np.zeros_like(expected)
    events = [
        onset(pitch=220).model_copy(update={"controls": {"level": 1}}),
        change(48000, 0.6, control="level"),
        Release(tick=84000, ordinal=0, part="main", trigger_id="note"),
    ]
    for channel, kind in enumerate(("synth", "sampler")):
        raw = lfo_score(kind).model_dump(mode="json")
        common = raw["body"] if kind == "synth" else raw["body"]["settings"]
        common["controls"] = {"level": {"default": 1}}
        voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
        voice["envelope"] = {
            "segments": [{"duration": "1/50", "target": 0.3}],
            "release": [{"duration": "1/4", "target": 0}],
        }
        voice["bindings"].append(
            {
                "name": "level",
                "kind": "control",
                "control": "level",
                "smoothing": "1/20",
            }
        )
        modulation = voice["modulation"]
        modulation["sources"].append(
            {"name": "level", "scope": "trigger", "minimum": 0, "maximum": 1}
        )
        modulation["parameters"].append(
            {
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "scope": "voice",
                "unit": "cents",
                "minimum": -60,
                "maximum": 60,
                "default": 0,
            }
        )
        modulation["routes"].extend(
            [
                {
                    "name": "level",
                    "source": "level",
                    "target": {"name": "processing", "parameter": "amplitude"},
                    "operation": "multiply",
                    "unit": "ratio",
                    "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
                },
                {
                    "name": "vibrato",
                    "source": "motion",
                    "target": {"name": "processing", "parameter": "tuning_cents"},
                    "operation": "add",
                    "unit": "cents",
                    "points": [
                        {"input": -1, "amount": -60},
                        {"input": 1, "amount": 60},
                    ],
                },
            ]
        )
        if kind == "synth":
            voice["oscillator"]["waveform"] = "sine"
            document = SynthInstrumentScore.model_validate(raw)
            actions = synth_trace.prepare(document.body, events, seed=0).actions
            renderer = synth.OfflineSynth(synth.prepare(document), backend)
        else:
            raw["assets"][0]["audio"]["frames"] = 120000
            raw["body"]["slices"][0]["end_frame"] = 120000
            voice["mapping"].update(pitch_tracking=True, reference_pitch_hz=220)
            document = instrument.SampleInstrumentScore.model_validate(raw)
            actions = trace.prepare(document.body, events, seed=0).actions
            renderer = sample_instrument.OfflineSampler(
                sample_instrument.prepare(
                    document, {"asset": np.column_stack([source, np.zeros(120000)])}
                ),
                backend,
            )
        actual[:, channel] = renderer.advance(actions, 0, length)[:, 0]
        assert renderer.snapshot().voices == []
    check_audio(tmp_path / "lfo-demo.wav", actual, expected)
    publish_flac(
        tmp_path / "lfo-demo-actual.wav",
        pytestconfig.cache.mkdir("audio") / f"lfo-demo-{backend}.flac",
    )
