from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import change, onset
from test_filters import matrix_filter
from test_sample_instrument import sample_score
from test_synth import check_audio, score
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import instrument, trace
from ufor.synth import SynthInstrumentScore

from enge import filters, sample_instrument, synth


def filter_score(kind: str) -> SynthInstrumentScore | instrument.SampleInstrumentScore:
    raw = (score() if kind == "synth" else sample_score()).model_dump(mode="json")
    common = raw["body"] if kind == "synth" else raw["body"]["settings"]
    common["controls"] = {"tone": {"default": 0}}
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    voice["processing"] = {
        "filters": [
            {
                "name": "tone",
                "response": "lowpass",
                "cutoff_hz": 500,
                "q": 1,
                "stages": 2,
            }
        ]
    }
    voice["envelope"] = {
        "segments": [{"duration": "1/100", "target": 0.2}],
        "release": [{"duration": "1/8", "target": 0}],
    }
    voice["lfos"] = {"motion": {"rate": "4", "scope": "voice"}}
    voice["bindings"] = [
        {"name": "tone", "kind": "control", "control": "tone", "smoothing": "1/200"},
        {"name": "motion", "kind": "lfo", "reference": "motion"},
    ]
    voice["modulation"] = {
        "sources": [
            {"name": "tone", "scope": "trigger", "minimum": 0, "maximum": 1},
            {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1},
        ],
        "parameters": [
            {
                "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                "unit": "hz",
                "scope": "voice",
                "minimum": 500,
                "maximum": 4000,
                "default": 500,
            },
            {
                "target": {"name": "filter-tone", "parameter": "q"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0.5,
                "maximum": 1.5,
                "default": 1,
            },
        ],
        "routes": [
            {
                "name": "cutoff",
                "source": "tone",
                "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                "operation": "add",
                "unit": "hz",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 3500}],
            },
            {
                "name": "resonance",
                "source": "motion",
                "target": {"name": "filter-tone", "parameter": "q"},
                "operation": "add",
                "unit": "ratio",
                "points": [{"input": -1, "amount": -0.5}, {"input": 1, "amount": 0.5}],
            },
        ],
    }
    if kind == "synth":
        voice["oscillator"]["waveform"] = "sine"
        return SynthInstrumentScore.model_validate(raw)
    voice["mapping"].update(pitch_tracking=False, reference_pitch_hz=None)
    # A duplicate local ID is a different filter in the instrument namespace.
    common["processing"] = {
        "filters": [
            {"name": "tone", "response": "highpass", "cutoff_hz": 300, "q": 0.7}
        ]
    }
    return instrument.SampleInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("kind", ["synth", "sampler"])
def test_controls_lfos_independent_voices_and_snapshots_drive_filters(
    kind: str,
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    document = filter_score(kind)
    events = [
        onset(pitch=220).model_copy(update={"controls": {"tone": 0}}),
        change(1001, 1, control="tone"),
        onset(6001, "second", pitch=330).model_copy(update={"controls": {"tone": 0.3}}),
        Release(tick=12001, ordinal=0, part="main", trigger_id="note"),
        change(13001, 0, control="tone"),
        Release(tick=18001, ordinal=0, part="main", trigger_id="second"),
    ]
    source = np.column_stack(
        [
            np.sin(2 * np.pi * np.arange(96000) * 220 / 48000),
            0.3 * np.cos(2 * np.pi * np.arange(96000) * 700 / 48000),
        ]
    )
    if isinstance(document, SynthInstrumentScore):
        prepared = synth.prepare(document)
        renderer = synth.OfflineSynth(prepared, backend)
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        definitions = document.body.voices[0].processing.filters
    else:
        prepared = sample_instrument.prepare(document, {"asset": source})
        renderer = sample_instrument.OfflineSampler(prepared, backend)
        actions = trace.prepare(document.body, events, seed=0).actions
        definitions = [
            *prepared.settings["sample"].processing.filters,
            *document.body.settings.processing.filters,
        ]
    chunks: list[np.ndarray] = []
    boundaries = sorted(
        {0, 1001, 1002, 6123, 13002, 24001, 48000, *range(0, 48000, 997)}
    )
    for start, end in pairwise(boundaries):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end in (1002, 6123, 13002):
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
    expected = np.zeros((48000, 2))
    for start, pitch, held, initial in ((0, 220, 12001, 0), (6001, 330, 12000, 0.3)):
        age = np.arange(held + 6000)
        tone = np.full(len(age), initial, dtype=np.float64)
        if start == 0:
            tone = np.clip((age - 1001) / 240, 0, 1) - np.clip(
                (age - 13001) / 240, 0, 1
            )
        values = filters.parameters(definitions, 48000, len(age))
        values[:, 0, 0] = 500 + 3500 * tone
        values[:, 0, 1] = 1 + 0.5 * np.sin(2 * np.pi * age / 12000)
        audio = (
            np.sin(2 * np.pi * age * pitch / 48000)[:, None]
            if kind == "synth"
            else source[: len(age)]
        )
        filtered = matrix_filter(definitions, audio, values)
        envelope = (
            0.2 * np.minimum(age / 480, 1) * np.clip(1 - (age - held) / 6000, 0, 1)
        )
        routed = np.column_stack(
            [
                filtered[:, 0],
                np.zeros(len(age))
                if kind == "synth"
                else 0.25 * filtered[:, 0] + filtered[:, 1],
            ]
        )
        expected[start : start + len(age)] += routed * envelope[:, None]
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "instrument-filters.wav", actual, expected)
    assert not np.any(actual[24001:])
    assert renderer.snapshot().voices == []


def test_sample_exhaustion_discards_filter_state_and_later_invalid_values(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    raw = filter_score("sampler").model_dump(mode="json")
    raw["assets"][0]["audio"]["frames"] = 2
    raw["body"]["slices"][0]["end_frame"] = 2
    voice = raw["body"]["slots"][0]
    voice["envelope"]["segments"] = [{"duration": "0", "target": 1}]
    voice["modulation"]["parameters"][0]["maximum"] = 30000
    voice["modulation"]["routes"][0]["points"][1]["amount"] = 29500
    voice["bindings"][0]["smoothing"] = "1/12000"
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(document, {"asset": np.ones((2, 2))})
    actions = trace.prepare(
        document.body,
        [
            onset().model_copy(update={"controls": {"tone": 0}}),
            change(0, 1, control="tone"),
        ],
        seed=0,
    ).actions
    renderer = sample_instrument.OfflineSampler(prepared, backend)
    actual = renderer.advance(actions, 0, 48000)
    definitions = [
        *prepared.settings["sample"].processing.filters,
        *document.body.settings.processing.filters,
    ]
    values = filters.parameters(definitions, 48000, 2)
    values[:, 0, 0] = [500, 7875]
    values[:, 0, 1] = 1 + 0.5 * np.sin(2 * np.pi * np.arange(2) / 12000)
    filtered = matrix_filter(definitions, np.ones((2, 2)), values)
    expected = np.zeros_like(actual)
    expected[:2] = filtered @ [[1, 0.25], [0, 1]]
    check_audio(tmp_path / "filter-exhaustion.wav", actual, expected)
    assert renderer.snapshot().voices == []


@pytest.mark.parametrize("kind", ["synth", "sampler"])
@pytest.mark.parametrize("boundary", ["clamp", "error"])
def test_live_cutoff_obeys_boundary_policy_before_processing_its_sample(
    kind: str,
    boundary: str,
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    raw = filter_score(kind).model_dump(mode="json")
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    voice["processing"]["filters"][0]["boundary"] = boundary
    voice["modulation"]["parameters"][0]["maximum"] = 30000
    voice["modulation"]["routes"][0]["points"][1]["amount"] = 29500
    voice["bindings"][0]["smoothing"] = "0"
    voice["envelope"] = {
        "segments": [{"duration": "0", "target": 1}],
        "release": [{"duration": "0", "target": 0}],
    }
    events = [
        onset(pitch=0.1).model_copy(update={"controls": {"tone": 0}}),
        change(500, 1, control="tone"),
        Release(tick=600, ordinal=0, part="main", trigger_id="note"),
    ]
    if kind == "synth":
        voice["oscillator"]["waveform"] = "square"
        document = SynthInstrumentScore.model_validate(raw)
        renderer = synth.OfflineSynth(synth.prepare(document), backend)
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        definitions = document.body.voices[0].processing.filters
    else:
        document = instrument.SampleInstrumentScore.model_validate(raw)
        prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
        renderer = sample_instrument.OfflineSampler(prepared, backend)
        actions = trace.prepare(document.body, events, seed=0).actions
        definitions = [
            *prepared.settings["sample"].processing.filters,
            *document.body.settings.processing.filters,
        ]
    if boundary == "error":
        with pytest.raises(ValueError, match="tone: invalid cutoff/Q at frame 500"):
            renderer.advance(actions, 0, 48000)
        return
    actual = renderer.advance(actions, 0, 48000)
    values = filters.parameters(definitions, 48000, 600)
    values[500:, 0, 0] = 23976
    values[:, 0, 1] = 1 + 0.5 * np.sin(2 * np.pi * np.arange(600) / 12000)
    expected = np.zeros_like(actual)
    signal = matrix_filter(definitions, np.ones((600, 1)), values)[:, 0]
    expected[:600, 0] = signal
    if kind == "sampler":
        expected[:600, 1] = 1.25 * signal
    check_audio(tmp_path / "clamped-filter.wav", actual, expected)
    assert renderer.snapshot().voices == []
