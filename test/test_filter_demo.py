from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import change, onset
from test_filter_instrument import filter_score
from test_filters import matrix_filter
from test_synth import check_audio
from test_synth_demo import publish_flac
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import instrument, trace
from ufor.synth import SynthInstrumentScore

from enge import filters, sample_instrument, synth


def test_filter_demo_sweeps_cutoff_and_modulates_resonance(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
    pytestconfig: pytest.Config,
) -> None:
    """Two seconds: filtered triangle synth left, sampled harmonic tone right."""
    age = np.arange(96000)
    cutoff = 500 + 3500 * (
        np.clip((age - 12000) / 6000, 0, 1)
        - 0.8 * np.clip((age - 36000) / 6000, 0, 1)
        + 0.6 * np.clip((age - 60000) / 6000, 0, 1)
    )
    phase = (age % 200) / 200
    triangle = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)
    harmonic = sum(
        np.sin(2 * np.pi * age * 240 * h / 48000) / h for h in (1, 2, 3, 5, 7)
    )
    envelope = 0.2 * np.minimum(age / 480, 1) * np.clip(1 - (age - 84000) / 6000, 0, 1)
    actual, expected = np.zeros((96000, 2)), np.zeros((96000, 2))
    events = [
        onset(pitch=240).model_copy(update={"controls": {"tone": 0}}),
        change(12000, 1, control="tone"),
        change(36000, 0.2, control="tone"),
        change(60000, 0.8, control="tone"),
        Release(tick=84000, ordinal=0, part="main", trigger_id="note"),
    ]
    for channel, kind in enumerate(("synth", "sampler")):
        raw = filter_score(kind).model_dump(mode="json")
        voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
        voice["bindings"][0]["smoothing"] = "1/8"
        if kind == "synth":
            voice["oscillator"]["waveform"] = "triangle"
            document = SynthInstrumentScore.model_validate(raw)
            renderer = synth.OfflineSynth(synth.prepare(document), backend)
            actions = synth_trace.prepare(document.body, events, seed=0).actions
            definitions = document.body.voices[0].processing.filters
        else:
            document = instrument.SampleInstrumentScore.model_validate(raw)
            prepared = sample_instrument.prepare(
                document, {"asset": np.column_stack([harmonic, np.zeros(96000)])}
            )
            renderer = sample_instrument.OfflineSampler(prepared, backend)
            actions = trace.prepare(document.body, events, seed=0).actions
            definitions = [
                *prepared.settings["sample"].processing.filters,
                *document.body.settings.processing.filters,
            ]
        actual[:, channel] = renderer.advance(actions, 0, 96000)[:, 0]
        values = filters.parameters(definitions, 48000, 90000)
        values[:, 0, 0] = cutoff[:90000]
        values[:, 0, 1] = 1 + 0.5 * np.sin(2 * np.pi * age[:90000] / 12000)
        source = triangle if kind == "synth" else harmonic
        expected[:90000, channel] = (
            matrix_filter(definitions, source[:90000, None], values)[:, 0]
            * envelope[:90000]
        )
        assert renderer.snapshot().voices == []
    check_audio(tmp_path / "filter-demo.wav", actual, expected)
    publish_flac(
        tmp_path / "filter-demo-actual.wav",
        pytestconfig.cache.mkdir("audio") / f"filter-demo-{backend}.flac",
    )
