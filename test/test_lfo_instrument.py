from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio, score
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import instrument, processing, trace
from ufor.synth import SynthInstrumentScore

from enge import sample_instrument, synth


def lfo_settings(scope: str = "voice") -> processing.SoundSettings:
    return processing.SoundSettings.model_validate(
        {
            "lfos": {
                "motion": {
                    "rate": "4",
                    "phase": "1/4",
                    "scope": scope,
                    "delay": "1/8",
                    "fade_in": "1/8",
                }
            },
            "bindings": [{"name": "motion", "kind": "lfo", "reference": "motion"}],
            "modulation": {
                "sources": [
                    {"name": "motion", "scope": scope, "minimum": -1, "maximum": 1}
                ],
                "parameters": [
                    {
                        "target": {"name": "processing", "parameter": "amplitude"},
                        "unit": "ratio",
                        "scope": "voice",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 1,
                    }
                ],
                "routes": [
                    {
                        "name": "tremolo",
                        "source": "motion",
                        "target": {"name": "processing", "parameter": "amplitude"},
                        "operation": "multiply",
                        "unit": "ratio",
                        "points": [
                            {"input": -1, "amount": 0.25},
                            {"input": 1, "amount": 1},
                        ],
                    }
                ],
            },
        }
    )


def lfo_score(
    kind: str, scope: str = "voice"
) -> SynthInstrumentScore | instrument.SampleInstrumentScore:
    raw = (score() if kind == "synth" else sample_score()).model_dump(mode="json")
    settings = lfo_settings(scope).model_dump(mode="json")
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    owner = raw["body"]["settings"] if kind == "sampler" and scope != "voice" else voice
    owner.update({n: settings[n] for n in ("lfos", "bindings", "modulation")})
    voice["envelope"] = {
        "segments": [{"duration": "0", "target": 1}],
        "release": [{"duration": "1/4", "target": 0}],
    }
    if kind == "synth":
        voice["oscillator"]["waveform"] = "square"
        return SynthInstrumentScore.model_validate(raw)
    voice["mapping"].update(pitch_tracking=False, reference_pitch_hz=None)
    return instrument.SampleInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_named_envelope_modulates_synth_and_releases(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = score().model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelope"]["release"] = [{"duration": "1/4", "target": 1}]
    voice.update(
        {
            "envelopes": {
                "motion": {
                    "initial": 0,
                    "segments": [{"duration": "1/4", "target": 1}],
                    "release": [{"duration": "1/4", "target": 0}],
                }
            },
            "bindings": [{"name": "motion", "kind": "envelope", "reference": "motion"}],
            "modulation": {
                "sources": [
                    {"name": "motion", "scope": "voice", "minimum": 0, "maximum": 1}
                ],
                "parameters": [
                    {
                        "target": {"name": "processing", "parameter": "amplitude"},
                        "unit": "ratio",
                        "scope": "voice",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 1,
                    }
                ],
                "routes": [
                    {
                        "name": "shape",
                        "source": "motion",
                        "target": {"name": "processing", "parameter": "amplitude"},
                        "operation": "multiply",
                        "unit": "ratio",
                        "points": [
                            {"input": 0, "amount": 0},
                            {"input": 1, "amount": 1},
                        ],
                    }
                ],
            },
        }
    )
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    renderer = synth.OfflineSynth(prepared, backend)
    actual = np.concatenate(
        [
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
            for start, end in ((0, 12000), (12000, 24000), (24000, 48000))
        ]
    )
    assert np.max(np.abs(actual[:100])) < np.max(np.abs(actual[11000:12000]))
    assert np.max(np.abs(actual[23000:])) < np.max(np.abs(actual[12000:13000]))
    check_audio(tmp_path / f"named-envelope-{backend}.wav", actual, actual)


@pytest.mark.parametrize("kind", ["synth", "sampler"])
@pytest.mark.parametrize("scope", ["voice", "part", "instrument"])
def test_lfo_scopes_silent_time_release_tails_and_restores(
    tmp_path: Path, backend: Literal["numpy", "native"], kind: str, scope: str
) -> None:
    document = lfo_score(kind, scope)
    events = [
        onset(6000, pitch=0.1).model_copy(update={"controls": {}}),
        onset(12000, "second", pitch=0.1).model_copy(update={"controls": {}}),
        onset(12000, "third", part="other", pitch=0.1).model_copy(
            update={"controls": {}, "ordinal": 1}
        ),
        Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        onset(24000, pitch=0.1).model_copy(update={"controls": {}}),
    ]
    if isinstance(document, SynthInstrumentScore):
        prepared = synth.prepare(document)
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        renderer = synth.OfflineSynth(prepared, backend)
    else:
        prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
        actions = trace.prepare(document.body, events, seed=0).actions
        renderer = sample_instrument.OfflineSampler(prepared, backend)
    chunks: list[np.ndarray] = []
    for start, end in pairwise(
        sorted(
            {0, 6000, 12000, 18000, 24000, 24001, 30000, 48000, *range(0, 48000, 997)}
        )
    ):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end in (12000, 24001):
            saved = renderer.snapshot().model_dump_json()
            if isinstance(prepared, synth.PreparedSynth):
                restored = synth.OfflineSynth(prepared, backend)
                restored.restore(synth.SynthSnapshot.model_validate_json(saved))
            else:
                restored = sample_instrument.OfflineSampler(prepared, backend)
                restored.restore(
                    sample_instrument.SamplerSnapshot.model_validate_json(saved)
                )
            renderer.advance([], end, end + 1)
            assert restored.snapshot().model_dump_json() == saved
            renderer = restored
    frames = np.arange(48000)
    expected_gain = np.zeros(48000)
    for start in (6000, 12000, 12000, 24000):
        age = frames - (start if scope == "voice" else 0)
        value = np.sin(2 * np.pi * (0.25 + age / 12000))
        weight = np.clip((age - 6000) / 6000, 0, 1)
        amplitude = 1 + weight * (0.625 + 0.375 * value - 1)
        envelope = np.clip(1 - (frames - 18000) / 12000, 0, 1) if start == 6000 else 1
        expected_gain += (frames >= start) * amplitude * envelope
    expected = np.column_stack(
        [expected_gain, np.zeros(48000) if kind == "synth" else 1.25 * expected_gain]
    )
    check_audio(tmp_path / "scoped-lfo.wav", np.concatenate(chunks), expected)
    assert (
        len(renderer.snapshot().lfos) == {"voice": 4, "part": 2, "instrument": 1}[scope]
    )


@pytest.mark.parametrize("kind", ["synth", "sampler"])
def test_named_lfo_names_remain_local_to_settings_and_instances(
    tmp_path: Path, backend: Literal["numpy", "native"], kind: str
) -> None:
    scope = "instrument" if kind == "synth" else "voice"
    raw = lfo_score(kind, scope).model_dump(mode="json")
    entries = raw["body"]["voices" if kind == "synth" else "slots"]
    entries.append(
        {
            **entries[0],
            "name": "second",
            "lfos": {"motion": {"rate": "0", "phase": "3/4", "scope": scope}},
        }
    )
    event = onset(pitch=0.1).model_copy(update={"controls": {}})
    if kind == "synth":
        document = SynthInstrumentScore.model_validate(raw)
        prepared = synth.prepare(document)
        actions = synth_trace.prepare(document.body, [event], seed=0).actions
        first = synth.OfflineSynth(prepared, backend)
        second = synth.OfflineSynth(prepared, backend)
    else:
        document = instrument.SampleInstrumentScore.model_validate(raw)
        prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
        actions = trace.prepare(document.body, [event], seed=0).actions
        first = sample_instrument.OfflineSampler(prepared, backend)
        second = sample_instrument.OfflineSampler(prepared, backend)
    actual = first.advance(actions, 0, 48000)
    frames = np.arange(48000)
    first_gain = 1 + np.clip((frames - 6000) / 6000, 0, 1) * (
        -0.375 + 0.375 * np.cos(2 * np.pi * frames / 12000)
    )
    gains = first_gain + 0.25
    expected = np.column_stack(
        [gains, np.zeros(48000) if kind == "synth" else gains * 1.25]
    )
    check_audio(tmp_path / "local-lfo-names.wav", actual, expected)
    assert second.snapshot().lfos == []
    np.testing.assert_allclose(
        second.advance(actions, 0, 48000), expected, atol=1e-10, rtol=1e-9
    )
