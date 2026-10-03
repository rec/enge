from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio, score
from ufor import instrument_trace, synth_trace
from ufor.control import TempoMap
from ufor.events import LFOChange, MotionChange, Release
from ufor.library import Entry, Library
from ufor.motion import (
    Cycle,
    MotionParameter,
    MotionScore,
    ParameterReference,
    advance_motion,
    initial_motion,
)
from ufor.samples import instrument, processing, trace
from ufor.synth import SynthInstrumentScore

from enge import sample_instrument, synth


def lfo_settings(scope: str = "voice") -> processing.SoundSettings:
    return processing.SoundSettings.model_validate(
        {
            "motions": {
                "motion": {
                    "scope": scope,
                    "body": {
                        "kind": "cycle",
                        "rate": "4",
                        "phase": "1/4",
                        "delay": "1/8",
                        "fade_in": "1/8",
                    },
                }
            },
            "bindings": [{"name": "motion", "kind": "motion", "reference": "motion"}],
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
    owner.update({n: settings[n] for n in ("motions", "bindings", "modulation")})
    voice["envelope"] = {
        "segments": [{"duration": "0 s", "to": 1}],
        "release": [{"duration": "1/4 s", "to": 0}],
    }
    if kind == "synth":
        voice["oscillator"]["waveform"] = "square"
        return SynthInstrumentScore.model_validate(raw)
    voice["mapping"].update(pitch_tracking=False, reference_pitch_hz=None)
    return instrument.SampleInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_beat_contour_follows_tempo_without_following_transport_seek(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"] = {
        "clock": "beats",
        "body": {
            "kind": "contour",
            "segments": [{"duration": "2 beats", "to": 1}],
        },
    }
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    definition = synth.prepare(document)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
                {"at_seconds": "1/2", "beat": "3/4", "bpm": "60", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    with pytest.raises(synth.EngineError, match="requires a host tempo map"):
        synth.OfflineSynth(definition, backend)
    with pytest.raises(synth.EngineError, match="requires a host tempo map"):
        synth.PersistentSynth(definition)
    renderer = synth.OfflineSynth(definition, backend, tempo_map=clock)
    first = renderer.advance(actions, 0, 24000)
    saved = renderer.snapshot()
    second = renderer.advance([], 24000, 48000)
    replay = synth.OfflineSynth(definition, backend, tempo_map=clock)
    replay.restore(synth.SynthSnapshot.model_validate_json(saved.model_dump_json()))
    np.testing.assert_array_equal(replay.advance([], 24000, 48000), second)
    with pytest.raises(synth.EngineError, match="different host tempo map"):
        synth.OfflineSynth(
            definition,
            backend,
            tempo_map=TempoMap.model_validate(
                {"points": [{"at_seconds": "0", "beat": "0", "bpm": "120"}]}
            ),
        ).restore(saved)
    seconds = np.arange(48000) / 48000
    beats = (
        2 * np.minimum(seconds, 0.25)
        + np.clip(seconds - 0.25, 0, 0.25)
        + 2 * np.maximum(seconds - 0.75, 0)
    )
    expected = np.zeros((48000, 2), dtype=np.float64)
    expected[:, 0] = 0.25 + 0.75 * beats / 2
    check_audio(
        tmp_path / f"beat-contour-{backend}.wav", np.vstack((first, second)), expected
    )
    persistent = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    native_first = persistent.advance(actions, 0, 24000)
    native_snapshot = persistent.snapshot()
    native_second = persistent.advance([], 24000, 48000)
    native_replay = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    native_replay.restore(native_snapshot)
    np.testing.assert_array_equal(
        native_replay.advance([], 24000, 48000), native_second
    )
    with pytest.raises(ValueError, match="different synth runtime"):
        synth.PersistentSynth(
            definition,
            voices=2,
            tempo_map=TempoMap.model_validate(
                {"points": [{"at_seconds": "0", "beat": "0", "bpm": "120"}]}
            ),
        ).restore(native_snapshot)
    check_audio(
        tmp_path / f"beat-contour-persistent-{backend}.wav",
        np.vstack((native_first, native_second)),
        expected,
    )


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_beat_stages_follow_tempo_and_stop_across_restore(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"] = {
        "clock": "beats",
        "body": {
            "kind": "stages",
            "initial_stage": "attack",
            "stages": [
                {
                    "name": "attack",
                    "motion": {
                        "kind": "contour",
                        "segments": [{"duration": "1 beat", "to": 1}],
                    },
                },
                {"name": "sustain", "motion": {"kind": "hold", "value": 1}},
            ],
            "transitions": [
                {
                    "from": ["attack"],
                    "event": "stage.done",
                    "action": {"kind": "enter", "stage": "sustain"},
                }
            ],
        },
    }
    document = SynthInstrumentScore.model_validate(raw)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
                {"at_seconds": "1/2", "beat": "3/4", "bpm": "60", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    with pytest.raises(synth.EngineError, match="requires a host tempo map"):
        synth.OfflineSynth(definition, backend)
    offline = synth.OfflineSynth(definition, backend, tempo_map=clock)
    first = offline.advance(actions, 0, 24000)
    snapshot = offline.snapshot()
    second = offline.advance([], 24000, 48000)
    replay = synth.OfflineSynth(definition, backend, tempo_map=clock)
    replay.restore(snapshot)
    np.testing.assert_array_equal(replay.advance([], 24000, 48000), second)
    seconds = np.arange(48000) / 48000
    beats = (
        2 * np.minimum(seconds, 0.25)
        + np.clip(seconds - 0.25, 0, 0.25)
        + 2 * np.maximum(seconds - 0.75, 0)
    )
    expected = np.zeros((48000, 2))
    expected[:, 0] = 0.625 + 0.375 * np.minimum(beats, 1)
    actual = np.vstack((first, second))
    check_audio(tmp_path / f"beat-stages-{backend}.wav", actual, expected)
    persistent = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    before = persistent.advance(actions, 0, 24000)
    saved = persistent.snapshot()
    after = persistent.advance([], 24000, 48000)
    resumed = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    resumed.restore(saved)
    np.testing.assert_array_equal(resumed.advance([], 24000, 48000), after)
    check_audio(
        tmp_path / "beat-stages-persistent.wav", np.vstack((before, after)), expected
    )


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_transport_position_cycle_follows_seek_without_crossing_history(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"] = {
        "clock": "beats",
        "position_driver": "transport",
        "body": {
            "kind": "cycle",
            "shape": "triangle",
            "rate": "1/4",
            "reset": "transport",
        },
    }
    document = SynthInstrumentScore.model_validate(raw)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
                {"at_seconds": "1/2", "beat": "3/4", "bpm": "60", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    seconds = np.arange(48000) / 48000
    song_beats = np.where(
        seconds < 0.25,
        2 * seconds,
        np.where(
            seconds < 0.5,
            0.5 + seconds - 0.25,
            np.where(seconds < 0.75, 0.75, 8 + 2 * (seconds - 0.75)),
        ),
    )
    phase = np.remainder(song_beats / 4, 1)
    shape = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)
    expected = np.zeros((48000, 2))
    expected[:, 0] = 0.625 + 0.375 * shape
    definition = synth.prepare(document)
    renderer = synth.OfflineSynth(definition, backend, tempo_map=clock)
    first = renderer.advance(actions, 0, 24000)
    saved = renderer.snapshot()
    second = renderer.advance([], 24000, 48000)
    restored = synth.OfflineSynth(definition, backend, tempo_map=clock)
    restored.restore(saved)
    np.testing.assert_array_equal(restored.advance([], 24000, 48000), second)
    check_audio(
        tmp_path / f"transport-cycle-{backend}.wav",
        np.vstack((first, second)),
        expected,
    )
    persistent = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    first = persistent.advance(actions, 0, 24000)
    saved = persistent.snapshot()
    second = persistent.advance([], 24000, 48000)
    restored = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
    restored.restore(saved)
    np.testing.assert_array_equal(restored.advance([], 24000, 48000), second)
    check_audio(
        tmp_path / "transport-cycle-persistent.wav",
        np.vstack((first, second)),
        expected,
    )


@pytest.mark.parametrize(
    ("shared_child", "shared_body"),
    [(False, "stages"), (True, "stages"), (True, "cycle"), (True, "contour")],
)
@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_patch_named_outputs_match_separate_motions(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    shared_child: bool,
    shared_body: str,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    stages = {
        "kind": "stages",
        "initial_stage": "attack",
        "stages": [
            {
                "name": "attack",
                "motion": {
                    "kind": "contour",
                    "segments": [{"duration": "1/4 s", "to": 1}],
                },
            },
            {
                "name": "sustain",
                "motion": {"kind": "cycle", "rate": "2", "center": 0.8, "depth": 0.2},
            },
            {
                "name": "release",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [{"duration": "1/4 s", "to": 0}],
                },
            },
        ],
        "transitions": [
            {
                "from": ["attack"],
                "event": "stage.done",
                "action": {"kind": "enter", "stage": "sustain"},
            },
            {
                "from": ["attack", "sustain"],
                "event": "note_off",
                "action": {"kind": "enter", "stage": "release"},
            },
        ],
    }
    vibrato = {"kind": "cycle", "rate": "5"}
    contour = {"kind": "contour", "segments": [{"duration": "1 s", "to": 1}]}
    child = {"stages": stages, "cycle": vibrato, "contour": contour}[shared_body]
    voice["motions"] = {"amp": {"body": child}}
    if not shared_child:
        voice["motions"]["vib"] = {"body": vibrato}
    voice["bindings"] = [
        {"name": "level", "kind": "motion", "reference": "amp"},
        {
            "name": "pitch",
            "kind": "motion",
            "reference": "amp" if shared_child else "vib",
        },
    ]
    voice["modulation"] = {
        "sources": [
            {
                "name": "level",
                "scope": "voice",
                "minimum": 0 if shared_child and shared_body == "contour" else -1,
                "maximum": 1,
            },
            {
                "name": "pitch",
                "scope": "voice",
                "minimum": 0 if shared_child and shared_body == "contour" else -1,
                "maximum": 1,
            },
        ],
        "parameters": [
            {
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
            },
            {
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "unit": "cents",
                "scope": "voice",
                "minimum": -100,
                "maximum": 100,
                "default": 0,
            },
        ],
        "routes": [
            {
                "name": "level",
                "source": "level",
                "target": {"name": "processing", "parameter": "amplitude"},
                "operation": "multiply",
                "unit": "ratio",
                "points": [
                    {
                        "input": 0 if shared_child and shared_body == "contour" else -1,
                        "amount": 0,
                    },
                    {"input": 1, "amount": 1},
                ],
            },
            {
                "name": "pitch",
                "source": "pitch",
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "operation": "add",
                "unit": "cents",
                "points": [
                    {
                        "input": 0 if shared_child and shared_body == "contour" else -1,
                        "amount": -50,
                    },
                    {"input": 1, "amount": 50},
                ],
            },
        ],
    }
    separate = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        separate.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(synth.prepare(separate), backend).advance(
        actions, 0, 48000
    )
    voice["motions"] = {
        "gesture": {
            "body": {
                "kind": "patch",
                "motions": {"amp": child}
                if shared_child
                else {"amp": stages, "vib": vibrato},
                "outputs": {"level": "amp", "pitch": "amp" if shared_child else "vib"},
            }
        }
    }
    voice["bindings"][0].update(reference="gesture", output="level")
    voice["bindings"][1].update(reference="gesture", output="pitch")
    patch = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(patch)
    patch_actions = synth_trace.prepare(
        patch.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    renderer = synth.OfflineSynth(prepared, backend)
    first_actions = [a for a in patch_actions if a.tick < 24000]
    second_actions = [a for a in patch_actions if a.tick >= 24000]
    first = renderer.advance(first_actions, 0, 24000)
    snapshot = renderer.snapshot()
    if shared_child and shared_body == "cycle":
        assert len(snapshot.lfos) == 1
    else:
        assert len(snapshot.envelopes) == 1
    second = renderer.advance(second_actions, 24000, 48000)
    restored = synth.OfflineSynth(prepared, backend)
    restored.restore(snapshot)
    np.testing.assert_array_equal(
        restored.advance(second_actions, 24000, 48000), second
    )
    actual = np.vstack((first, second))
    check_audio(tmp_path / f"patch-{backend}.wav", actual, expected)
    live = synth.PersistentSynth(prepared, voices=2)
    first = live.advance(first_actions, 0, 24000)
    snapshot = live.snapshot()
    second = live.advance(second_actions, 24000, 48000)
    replay = synth.PersistentSynth(prepared, voices=2)
    replay.restore(snapshot)
    np.testing.assert_array_equal(replay.advance(second_actions, 24000, 48000), second)
    check_audio(tmp_path / "patch-persistent.wav", np.vstack((first, second)), expected)
    command = instrument_trace.MotionObservation(
        tick=12000,
        ordinal=0,
        name="gesture",
        part="main",
        trigger_id="note",
        action="seek",
        position=0.5,
    )
    with pytest.raises(synth.EngineError, match="Patch-level Motion commands"):
        synth.OfflineSynth(prepared, backend).advance(
            sorted([*patch_actions, command], key=lambda a: (a.tick, a.ordinal)),
            0,
            48000,
        )
    with pytest.raises(synth.EngineError, match="Patch-level Motion commands"):
        synth.PersistentSynth(prepared, voices=2).advance(
            sorted([*patch_actions, command], key=lambda a: (a.tick, a.ordinal)),
            0,
            48000,
        )


@pytest.mark.parametrize(("phase", "first_marker"), [("0", 6000), ("1/7", 2572)])
def test_patch_cycle_marker_starts_unexposed_child_contour(
    tmp_path: Path, phase: str, first_marker: int
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "gesture": {
            "body": {
                "kind": "patch",
                "motions": {
                    "clock": {
                        "kind": "cycle",
                        "rate": "2",
                        "phase": phase,
                        "markers": [{"name": "peak", "position": "1/4"}],
                    },
                    "accent": {
                        "kind": "contour",
                        "start": "event",
                        "segments": [
                            {"duration": "1/8 s", "to": 1},
                            {"duration": "1/8 s", "to": 0},
                        ],
                    },
                },
                "events": [
                    {"source": "clock.peak", "target": "accent", "action": "start"}
                ],
                "outputs": {"value": "accent"},
            }
        }
    }
    voice["motions"]["unused"] = voice["motions"]["gesture"]
    voice["bindings"] = [
        {"name": "level", "kind": "motion", "reference": "gesture", "output": "value"}
    ]
    voice["modulation"] = {
        "sources": [{"name": "level", "scope": "voice", "minimum": 0, "maximum": 1}],
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
                "name": "accent",
                "source": "level",
                "target": {"name": "processing", "parameter": "amplitude"},
                "operation": "multiply",
                "unit": "ratio",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body, [onset(0, pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    offline = synth.OfflineSynth(prepared).advance(actions, 0, 48000)
    assert np.max(np.abs(offline[:first_marker])) == 0
    assert np.max(np.abs(offline[first_marker + 6000 : first_marker + 9000])) > 0.1
    assert np.max(np.abs(offline[first_marker + 23999 : first_marker + 24000])) == 0
    for backend in ("numpy", "native"):
        renderer = synth.OfflineSynth(prepared, backend)
        first = renderer.advance(actions, 0, first_marker)
        snapshot = renderer.snapshot()
        second = renderer.advance([], first_marker, 48000)
        restored = synth.OfflineSynth(prepared, backend)
        restored.restore(snapshot)
        np.testing.assert_array_equal(restored.advance([], first_marker, 48000), second)
        check_audio(
            tmp_path / f"patch-events-{backend}-{phase.replace('/', '-')}.wav",
            np.vstack((first, second)),
            offline,
        )
    live = synth.PersistentSynth(prepared, voices=2)
    first = live.advance(actions, 0, 24000)
    snapshot = live.snapshot()
    second = live.advance([], 24000, 48000)
    restored = synth.PersistentSynth(prepared, voices=2)
    restored.restore(snapshot)
    np.testing.assert_array_equal(restored.advance([], 24000, 48000), second)
    check_audio(
        tmp_path / f"patch-events-persistent-{phase.replace('/', '-')}.wav",
        np.vstack((first, second)),
        offline,
    )


@pytest.mark.parametrize(
    ("source_kind", "phase", "first_event"),
    [
        ("cycle", "0", 6000),
        ("cycle", "1/7", 2572),
        ("staged-marker", "0", 6000),
        ("staged-marker", "1/7", 2572),
        ("staged-done", "0", 12000),
    ],
)
def test_patch_child_event_cues_child_stages(
    tmp_path: Path, source_kind: str, phase: str, first_event: int
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    clock = {
        "kind": "cycle",
        "rate": "2",
        "phase": phase,
        "markers": [{"name": "peak", "position": "1/4"}],
    }
    if source_kind != "cycle":
        clock = {
            "kind": "stages",
            "initial_stage": "playing",
            "stages": [
                {
                    "name": "playing",
                    "motion": clock
                    if source_kind == "staged-marker"
                    else {
                        "kind": "contour",
                        "segments": [{"duration": "1/4 s", "to": 1}],
                    },
                }
            ],
        }
    voice["motions"] = {
        "gesture": {
            "body": {
                "kind": "patch",
                "motions": {
                    "clock": clock,
                    "level": {
                        "kind": "stages",
                        "initial_stage": "waiting",
                        "stages": [
                            {
                                "name": "waiting",
                                "motion": {"kind": "hold", "value": -1},
                            },
                            {"name": "bright", "motion": {"kind": "hold", "value": 1}},
                        ],
                        "transitions": [
                            {
                                "from": ["waiting"],
                                "event": "cue.brighten",
                                "action": {"kind": "enter", "stage": "bright"},
                            }
                        ],
                    },
                },
                "outputs": {"value": "level"},
                "events": [
                    {
                        "source": "clock.stage.done"
                        if source_kind == "staged-done"
                        else "clock.peak",
                        "target": "level",
                        "action": "cue",
                        "cue": "brighten",
                    }
                ],
            }
        }
    }
    voice["bindings"] = [
        {"name": "level", "kind": "motion", "reference": "gesture", "output": "value"}
    ]
    voice["modulation"]["sources"] = [
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1}
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body, [onset(0, pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    reference = synth.OfflineSynth(prepared).advance(actions, 0, 48000)
    assert np.max(np.abs(reference[:first_event])) == 0
    assert reference[first_event, 0] == pytest.approx(1)
    for backend in ("numpy", "native"):
        renderer = synth.OfflineSynth(prepared, backend)
        first = renderer.advance(actions, 0, 16000)
        snapshot = renderer.snapshot()
        second = renderer.advance([], 16000, 48000)
        restored = synth.OfflineSynth(prepared, backend)
        restored.restore(snapshot)
        np.testing.assert_array_equal(restored.advance([], 16000, 48000), second)
        check_audio(
            tmp_path
            / f"patch-cue-{source_kind}-{backend}-{phase.replace('/', '-')}.wav",
            np.vstack((first, second)),
            reference,
        )
    live = synth.PersistentSynth(prepared, voices=2)
    first = live.advance(actions, 0, 16000)
    snapshot = live.snapshot()
    second = live.advance([], 16000, 48000)
    restored = synth.PersistentSynth(prepared, voices=2)
    restored.restore(snapshot)
    np.testing.assert_array_equal(restored.advance([], 16000, 48000), second)
    check_audio(
        tmp_path / f"patch-cue-{source_kind}-persistent-{phase.replace('/', '-')}.wav",
        np.vstack((first, second)),
        reference,
    )


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_beat_cycle_tracks_host_tempo_in_both_offline_backends(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"] = {
        "clock": "beats",
        "body": {
            "kind": "cycle",
            "rate": "1",
            "delay": "1/4",
            "fade_in": "1/4",
        },
    }
    document = SynthInstrumentScore.model_validate(raw)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
                {"at_seconds": "1/2", "beat": "3/4", "bpm": "60", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    actual = synth.OfflineSynth(
        synth.prepare(document), backend, tempo_map=clock
    ).advance(actions, 0, 48000)
    seconds = np.arange(48000) / 48000
    beats = (
        2 * np.minimum(seconds, 0.25)
        + np.clip(seconds - 0.25, 0, 0.25)
        + 2 * np.maximum(seconds - 0.75, 0)
    )
    expected = np.zeros((48000, 2), dtype=np.float64)
    weight = np.clip((beats - 0.25) / 0.25, 0, 1)
    expected[:, 0] = 1 + weight * (0.625 + 0.375 * np.sin(2 * np.pi * beats) - 1)
    check_audio(tmp_path / f"beat-cycle-{backend}.wav", actual, expected)
    if backend == "numpy":
        native = synth.PersistentSynth(
            synth.prepare(document), voices=2, tempo_map=clock
        ).advance(actions, 0, 48000)
        check_audio(tmp_path / "beat-cycle-persistent.wav", native, expected)


def test_persistent_beat_contour_releases_at_frame_boundary_while_clock_stops(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelope"]["release"] = [{"duration": "1 s", "to": 0}]
    voice["motions"]["motion"] = {
        "clock": "beats",
        "body": {
            "kind": "contour",
            "initial": 1,
            "segments": [{"duration": "1 beat", "to": 1}],
            "release": [{"duration": "1 beat", "to": 0}],
        },
    }
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    definition = synth.prepare(document)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/2", "beat": "1", "bpm": "120", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=24000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(definition, tempo_map=clock).advance(
        actions, 0, 48000
    )
    actual = synth.PersistentSynth(definition, voices=2, tempo_map=clock).advance(
        actions, 0, 48000
    )
    check_audio(tmp_path / "beat-contour-stop-release.wav", actual, expected)


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("scope", ["voice", "instrument"])
def test_beat_cycle_pause_preserves_musical_phase(
    tmp_path: Path, backend: Literal["numpy", "native"], scope: str
) -> None:
    raw = lfo_score("synth", scope).model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"] = {
        "scope": scope,
        "clock": "beats",
        "body": {"kind": "cycle", "rate": "1"},
    }
    document = SynthInstrumentScore.model_validate(raw)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/2", "beat": "1", "bpm": "60"},
            ]
        }
    )
    changes = (
        [
            LFOChange(tick=12000, ordinal=0, name="motion", action="pause"),
            LFOChange(tick=24000, ordinal=0, name="motion", action="resume"),
        ]
        if scope == "instrument"
        else [
            MotionChange(
                tick=12000,
                ordinal=0,
                name="motion",
                part="main",
                trigger_id="note",
                action="pause",
            ),
            MotionChange(
                tick=24000,
                ordinal=0,
                name="motion",
                part="main",
                trigger_id="note",
                action="resume",
            ),
        ]
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}}), *changes],
        seed=0,
    ).actions
    actual = synth.OfflineSynth(
        synth.prepare(document), backend, tempo_map=clock
    ).advance(actions, 0, 48000)
    seconds = np.arange(48000) / 48000
    beats = 2 * np.minimum(seconds, 0.25) + np.maximum(seconds - 0.5, 0)
    expected = np.zeros((48000, 2), dtype=np.float64)
    expected[:, 0] = 0.625 + 0.375 * np.sin(2 * np.pi * beats)
    check_audio(tmp_path / f"beat-cycle-pause-{backend}-{scope}.wav", actual, expected)
    if backend == "numpy":
        definition = synth.prepare(document)
        native = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
        first = native.advance([a for a in actions if a.tick < 24000], 0, 24000)
        saved = native.snapshot()
        second = native.advance([a for a in actions if a.tick >= 24000], 24000, 48000)
        replay = synth.PersistentSynth(definition, voices=2, tempo_map=clock)
        replay.restore(saved)
        np.testing.assert_array_equal(
            replay.advance([a for a in actions if a.tick >= 24000], 24000, 48000),
            second,
        )
        check_audio(
            tmp_path / f"beat-cycle-pause-persistent-{scope}.wav",
            np.vstack((first, second)),
            expected,
        )


@pytest.mark.parametrize("body_kind", ["contour", "stages"])
@pytest.mark.parametrize("kind", ["synth", "sampler"])
def test_trigger_addressed_motion_changes_only_matching_voice(
    tmp_path: Path, body_kind: str, kind: str
) -> None:
    raw = lfo_score(kind, "voice").model_dump(mode="json")
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    contour = {"kind": "contour", "segments": [{"duration": "1 s", "to": 1}]}
    body = (
        contour
        if body_kind == "contour"
        else {
            "kind": "stages",
            "initial_stage": "rise",
            "stages": [{"name": "rise", "motion": contour}],
        }
    )
    voice["motions"]["motion"]["body"] = body
    if body_kind == "contour":
        modulation = voice["modulation"]
        modulation["sources"][0]["minimum"] = 0
        modulation["routes"][0]["points"][0]["input"] = 0
    document = (
        SynthInstrumentScore.model_validate(raw)
        if kind == "synth"
        else instrument.SampleInstrumentScore.model_validate(raw)
    )
    first = onset(0, pitch=0.1).model_copy(update={"controls": {}})
    second = onset(0, "second", pitch=0.2).model_copy(
        update={"controls": {}, "ordinal": 1}
    )
    changes = [
        MotionChange(
            tick=12000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="pause",
        ),
        MotionChange(
            tick=24000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="seek",
            position=0.75,
        ),
        MotionChange(
            tick=36000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="resume",
        ),
        MotionChange(
            tick=40000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=-0.5,
        ),
    ]
    definition = (
        synth.prepare(document)
        if isinstance(document, SynthInstrumentScore)
        else sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
    )

    def render(events: list[object]) -> np.ndarray:
        if isinstance(document, SynthInstrumentScore):
            assert isinstance(definition, synth.PreparedSynth)
            actions = synth_trace.prepare(document.body, events, seed=0).actions
            return synth.OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
        assert isinstance(definition, sample_instrument.PreparedSampler)
        actions = trace.prepare(document.body, events, seed=0).actions
        return sample_instrument.OfflineSampler(definition, "numpy").advance(
            actions, 0, 48000
        )

    actual = render([first, second, *changes])
    expected = render([first, *changes]) + render([second])
    check_audio(tmp_path / f"trigger-{kind}-{body_kind}.wav", actual, expected)
    assert not np.allclose(actual, render([first, second]))


@pytest.mark.parametrize("playback", ["loop", "ping_pong"])
def test_sampler_contour_playback_and_release_survive_restore(
    tmp_path: Path, playback: str
) -> None:
    raw = lfo_score("sampler", "voice").model_dump(mode="json")
    slot = raw["body"]["slots"][0]
    slot["motions"]["motion"]["body"] = {
        "kind": "contour",
        "playback": playback,
        "segments": [{"duration": "1/2 s", "to": 1}],
        "release": [{"duration": "1/4 s", "to": 0}],
    }
    slot["modulation"]["sources"][0]["minimum"] = 0
    slot["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = instrument.SampleInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=26000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=1.5,
        ),
        Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = trace.prepare(document.body, events, seed=0).actions
    definition = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    renderer = sample_instrument.OfflineSampler(definition)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        if end == 24925:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / f"sampler-contour-{playback}.wav", actual, expected)
    restored = sample_instrument.OfflineSampler(definition)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 24925 <= a.tick < 48000], 24925, 48000
    )
    np.testing.assert_allclose(replay, actual[24925:], atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_trigger_motion_change_reaches_every_started_voice(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    single = lfo_score("synth", "voice")
    assert isinstance(single, SynthInstrumentScore)
    raw = single.model_dump(mode="json")
    layer = raw["body"]["voices"][0].copy()
    layer["name"] = "second-layer"
    raw["body"]["voices"].append(layer)
    layered = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=12000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="pause",
        ),
    ]
    expected_actions = synth_trace.prepare(single.body, events, seed=0).actions
    actual_actions = synth_trace.prepare(layered.body, events, seed=0).actions
    expected = 2 * synth.OfflineSynth(synth.prepare(single), backend).advance(
        expected_actions, 0, 48000
    )
    actual = synth.OfflineSynth(synth.prepare(layered), backend).advance(
        actual_actions, 0, 48000
    )
    check_audio(tmp_path / f"trigger-layers-{backend}.wav", actual, expected)


def test_library_motion_materializes_before_synth_preparation(tmp_path: Path) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"] = {
        "scope": "voice",
        "score": {"selector": "motions:vibrato"},
        "parameters": {"speed": 6},
    }
    authored = SynthInstrumentScore.model_validate(raw)
    with pytest.raises(synth.EngineError, match="materialized before rendering"):
        synth.prepare(authored)
    library = Library(
        [
            Entry(
                library="motions",
                address="/vibrato.toml",
                name="vibrato",
                sha256="c" * 64,
                score=MotionScore(
                    name="vibrato",
                    title="Vibrato",
                    parameters={
                        "speed": MotionParameter(
                            unit="hz", default=5, minimum=0.1, maximum=20
                        )
                    },
                    body=Cycle(
                        rate=ParameterReference(parameter="speed"),
                        phase=Fraction(1, 4),
                        delay=Fraction(1, 8),
                        fade_in=Fraction(1, 8),
                    ),
                ),
            ),
            Entry(
                library="motions", address="/patch.toml", name="patch", score=authored
            ),
        ]
    )
    materialized = library.materialize("motions:patch")
    assert isinstance(materialized, SynthInstrumentScore)
    inline = lfo_score("synth").model_dump(mode="json")
    inline["body"]["voices"][0]["motions"]["motion"]["body"]["rate"] = "6"
    expected = SynthInstrumentScore.model_validate(inline)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(
        materialized.body,
        events,
        seed=0,
    ).actions
    expected_actions = synth_trace.prepare(
        expected.body,
        events,
        seed=0,
    ).actions
    renderer = synth.OfflineSynth(synth.prepare(materialized))
    actual_audio = renderer.advance(actions, 0, 48000)
    expected_audio = synth.OfflineSynth(synth.prepare(expected)).advance(
        expected_actions, 0, 48000
    )
    check_audio(tmp_path / "library-vibrato.wav", actual_audio, expected_audio)
    origin = (
        renderer.snapshot().definition.instrument.voices[0].motions["motion"].origin
    )
    assert origin is not None
    assert origin.identity == "motions:/vibrato.toml"
    assert origin.sha256 == "c" * 64


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_named_envelope_modulates_synth_and_releases(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = score().model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelope"]["release"] = [{"duration": "1/4 s", "to": 1}]
    voice.update(
        {
            "motions": {
                "motion": {
                    "body": {
                        "kind": "contour",
                        "initial": 0,
                        "segments": [{"duration": "1/4 s", "to": 1}],
                        "release": [{"duration": "1/4 s", "to": 0}],
                    }
                }
            },
            "bindings": [{"name": "motion", "kind": "motion", "reference": "motion"}],
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
    if backend == "native":
        persistent = synth.PersistentSynth(prepared, voices=4)
        persistent_actual = np.empty_like(actual)
        for start, end in ((0, 997), (997, 12000), (12000, 24925), (24925, 48000)):
            persistent_actual[start:end] = persistent.advance(
                [action for action in actions if start <= action.tick < end],
                start,
                end,
            )
            if end == 24925:
                snapshot = persistent.snapshot()
        restored = synth.PersistentSynth(prepared, voices=4)
        restored.restore(snapshot)
        replay = restored.advance(
            [action for action in actions if 24925 <= action.tick < 48000],
            24925,
            48000,
        )
        check_audio(
            tmp_path / "persistent-named-envelope.wav", persistent_actual, actual
        )
        np.testing.assert_allclose(replay, persistent_actual[24925:], atol=0)


@pytest.mark.parametrize("release_timing", ["event", "voice"])
@pytest.mark.parametrize("hold", ["1/2", "48001/96000"])
@pytest.mark.parametrize("release_duration", ["1/4 s", "0 s"])
def test_named_contour_releases_before_voice_minimum_hold(
    tmp_path: Path,
    release_timing: Literal["event", "voice"],
    hold: str,
    release_duration: str,
) -> None:
    raw = score().model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["minimum_hold_seconds"] = hold
    voice["envelope"]["release"] = [{"duration": "1/4 s", "to": 0}]
    voice["motions"] = {
        "motion": {
            "body": {
                "kind": "contour",
                "initial": 1,
                "segments": [{"duration": "1/4 s", "to": 1}],
                "release": [{"duration": release_duration, "to": 0}],
            }
        }
    }
    voice["bindings"] = [
        {
            "name": "motion",
            "kind": "motion",
            "reference": "motion",
            "release_timing": release_timing,
        }
    ]
    voice["modulation"] = {
        "sources": [{"name": "motion", "scope": "voice", "minimum": 0, "maximum": 1}],
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
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=4800, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    retirement = next(
        a for a in actions if isinstance(a, instrument_trace.VoiceRetirement)
    )
    actions.append(retirement.model_copy(update={"tick": 9600}))
    reference = synth.OfflineSynth(prepared, "numpy").advance(actions, 0, 48000)
    offline = synth.OfflineSynth(prepared, "native")
    offline_first = offline.advance(actions, 0, 20000)
    offline_snapshot = offline.snapshot()
    offline_second = offline.advance([], 20000, 48000)
    offline_replay = synth.OfflineSynth(prepared, "native")
    offline_replay.restore(offline_snapshot)
    np.testing.assert_allclose(offline_replay.advance([], 20000, 48000), offline_second)
    check_audio(
        tmp_path / f"named-contour-minimum-hold-{release_timing}.wav",
        np.concatenate((offline_first, offline_second)),
        reference,
    )
    persistent = synth.PersistentSynth(prepared, voices=4)
    first = persistent.advance(actions, 0, 20000)
    snapshot = persistent.snapshot()
    second = persistent.advance([], 20000, 48000)
    restored = synth.PersistentSynth(prepared, voices=4)
    restored.restore(snapshot)
    np.testing.assert_allclose(restored.advance([], 20000, 48000), second, atol=0)
    check_audio(
        tmp_path / f"persistent-contour-minimum-hold-{release_timing}.wav",
        np.concatenate((first, second)),
        reference,
    )
    assert (np.max(np.abs(reference[18000:20000])) > 0.01) == (
        release_timing == "voice"
    )
    if release_timing == "voice":
        stopped_actions = [
            *actions,
            retirement.model_copy(update={"tick": 12000, "action": "stop"}),
        ]
        stopped = synth.OfflineSynth(prepared, "native")
        stopped_audio = stopped.advance(stopped_actions, 0, 48000)
        assert all(s.pending_release is None for s in stopped.snapshot().envelopes)
        check_audio(
            tmp_path / "stopped-contour-minimum-hold.wav",
            synth.PersistentSynth(prepared, voices=4).advance(
                stopped_actions, 0, 48000
            ),
            stopped_audio,
        )
        assert np.all(stopped_audio[12000:] == 0)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_release_free_contour_finishes_after_note_off_and_restores(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelope"]["release"] = [{"duration": "1 s", "to": 1}]
    voice["motions"]["motion"]["body"] = {
        "kind": "contour",
        "initial": 0,
        "segments": [{"duration": "1/2 s", "to": 1}],
    }
    source = voice["modulation"]["sources"][0]
    source["minimum"] = 0
    route = voice["modulation"]["routes"][0]
    route["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    prepared = synth.prepare(document)
    expected = synth.OfflineSynth(prepared, backend).advance(actions, 0, 48000)
    renderer = synth.OfflineSynth(prepared, backend)
    first = renderer.advance([a for a in actions if a.tick < 18000], 0, 18000)
    snapshot = renderer.snapshot()
    second = renderer.advance([], 18000, 48000)
    restored = synth.OfflineSynth(prepared, backend)
    restored.restore(snapshot)
    replay = restored.advance([], 18000, 48000)
    actual = np.concatenate([first, second])
    check_audio(tmp_path / f"one-shot-contour-{backend}.wav", actual, expected)
    np.testing.assert_allclose(replay, second, atol=0)
    assert np.max(np.abs(actual[23000:24000])) > np.max(np.abs(actual[11000:12000]))
    assert np.max(np.abs(actual[36000:37000])) == pytest.approx(
        np.max(np.abs(actual[24000:25000]))
    )
    if backend == "native":
        reference = synth.OfflineSynth(prepared, "numpy").advance(actions, 0, 48000)
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-9)
        persistent = synth.PersistentSynth(prepared, voices=4)
        chunks = []
        for start, end in ((0, 997), (997, 12000), (12000, 18000), (18000, 48000)):
            chunks.append(
                persistent.advance(
                    [a for a in actions if start <= a.tick < end], start, end
                )
            )
            if end == 18000:
                native_snapshot = persistent.snapshot()
        native_actual = np.concatenate(chunks)
        native_restored = synth.PersistentSynth(prepared, voices=4)
        native_restored.restore(native_snapshot)
        native_replay = native_restored.advance([], 18000, 48000)
        check_audio(
            tmp_path / "one-shot-contour-persistent.wav", native_actual, expected
        )
        np.testing.assert_allclose(native_replay, native_actual[18000:], atol=0)


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("release_frame", [12000, 12001, 24000])
def test_staged_motion_renders_and_restores_at_attack_boundary(
    tmp_path: Path, backend: Literal["numpy", "native"], release_frame: int
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelope"]["release"] = [{"duration": "1 s", "to": 1}]
    voice["motions"]["motion"]["body"] = {
        "kind": "stages",
        "initial_stage": "waiting",
        "stages": [
            {"name": "waiting", "motion": {"kind": "hold", "value": 0.0}},
            {
                "name": "attack",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [
                        {
                            "duration": "24001/96000 s"
                            if release_frame == 12001
                            else "1/4 s",
                            "to": 0.6,
                        }
                    ],
                },
            },
            {
                "name": "sway",
                "motion": {
                    "kind": "cycle",
                    "rate": "4",
                    "center": 0.6,
                    "depth": 0.2,
                },
            },
            {
                "name": "release",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [{"duration": "1/4 s", "to": 0.0}],
                },
            },
        ],
        "transitions": [
            {
                "from": ["waiting", "attack", "sway", "release"],
                "event": "note_on",
                "action": {"kind": "enter", "stage": "attack"},
            },
            {
                "from": ["attack"],
                "event": "stage.done",
                "action": {"kind": "enter", "stage": "sway"},
            },
            {
                "from": ["attack", "sway"],
                "event": "note_off",
                "action": {"kind": "enter", "stage": "release"},
            },
            {
                "from": ["release"],
                "event": "stage.done",
                "action": {"kind": "finish"},
            },
        ],
    }
    voice["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=release_frame, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(prepared, backend).advance(actions, 0, 48000)
    renderer = synth.OfflineSynth(prepared, backend)
    first = renderer.advance([a for a in actions if a.tick < 18000], 0, 18000)
    snapshot = renderer.snapshot()
    second = renderer.advance([a for a in actions if a.tick >= 18000], 18000, 48000)
    restored = synth.OfflineSynth(prepared, backend)
    restored.restore(snapshot)
    replay = restored.advance([a for a in actions if a.tick >= 18000], 18000, 48000)
    check_audio(
        tmp_path / f"staged-{backend}-{release_frame}.wav",
        np.concatenate((first, second)),
        expected,
    )
    np.testing.assert_allclose(replay, second, atol=0)
    assert snapshot.envelopes[0].state.runtime.stage == (
        "release" if release_frame < 24000 else "sway"
    )
    if release_frame < 24000:
        assert np.max(np.abs(expected[12000:13000])) > np.max(
            np.abs(expected[23000:24000])
        )
    else:
        assert np.ptp(expected[12000:24000, 0]) > 0.19
    if backend == "native":
        persistent = synth.PersistentSynth(prepared, voices=4)
        native_first = persistent.advance(
            [a for a in actions if a.tick < 18000], 0, 18000
        )
        native_snapshot = persistent.snapshot()
        native_second = persistent.advance(
            [a for a in actions if a.tick >= 18000], 18000, 48000
        )
        native_restored = synth.PersistentSynth(prepared, voices=4)
        native_restored.restore(native_snapshot)
        native_replay = native_restored.advance(
            [a for a in actions if a.tick >= 18000], 18000, 48000
        )
        check_audio(
            tmp_path / f"persistent-staged-{release_frame}.wav",
            np.concatenate((native_first, native_second)),
            expected,
        )
        np.testing.assert_allclose(native_replay, native_second, atol=0)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_staged_sampler_motion_releases_and_restores(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = lfo_score("sampler").model_dump(mode="json")
    slot = raw["body"]["slots"][0]
    slot["envelope"]["release"] = [{"duration": "1 s", "to": 1}]
    slot["motions"]["motion"]["body"] = {
        "kind": "stages",
        "initial_stage": "waiting",
        "stages": [
            {"name": "waiting", "motion": {"kind": "hold", "value": 0.0}},
            {
                "name": "attack",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [{"duration": "1/4 s", "to": 1.0}],
                },
            },
            {
                "name": "release",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [{"duration": "1/4 s", "to": 0.0}],
                },
            },
        ],
        "transitions": [
            {
                "from": ["waiting"],
                "event": "note_on",
                "action": {"kind": "enter", "stage": "attack"},
            },
            {
                "from": ["attack"],
                "event": "note_off",
                "action": {"kind": "enter", "stage": "release"},
            },
            {
                "from": ["release"],
                "event": "stage.done",
                "action": {"kind": "finish"},
            },
        ],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
    actions = trace.prepare(
        document.body,
        [
            onset(0, pitch=440).model_copy(update={"controls": {}}),
            Release(tick=24000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    expected = sample_instrument.OfflineSampler(prepared, backend).advance(
        actions, 0, 48000
    )
    renderer = sample_instrument.OfflineSampler(prepared, backend)
    first = renderer.advance([a for a in actions if a.tick < 30000], 0, 30000)
    snapshot = renderer.snapshot()
    second = renderer.advance([], 30000, 48000)
    restored = sample_instrument.OfflineSampler(prepared, backend)
    restored.restore(
        sample_instrument.SamplerSnapshot.model_validate_json(
            snapshot.model_dump_json()
        )
    )
    replay = restored.advance([], 30000, 48000)
    check_audio(
        tmp_path / f"staged-sampler-{backend}.wav",
        np.concatenate((first, second)),
        expected,
    )
    np.testing.assert_allclose(replay, second, atol=0)
    assert snapshot.envelopes[0].state.runtime.stage == "release"
    assert expected[24000, 0] > expected[36000, 0]


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("cascade", [False, True])
@pytest.mark.parametrize("position", ["1/4", "24001/96000"])
def test_staged_marker_cues_another_motion_at_exact_time(
    tmp_path: Path, backend: Literal["numpy", "native"], cascade: bool, position: str
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "clock": {
            "body": {
                "kind": "stages",
                "initial_stage": "cycle",
                "stages": [
                    {
                        "name": "cycle",
                        "motion": {
                            "kind": "cycle",
                            "rate": "1",
                            "markers": [{"name": "peak", "position": position}],
                        },
                    }
                ],
            }
        },
        "level": {
            "body": {
                "kind": "stages",
                "initial_stage": "waiting",
                "stages": [
                    {"name": "waiting", "motion": {"kind": "hold", "value": 0.0}},
                    {"name": "bright", "motion": {"kind": "hold", "value": 1.0}},
                ],
                "transitions": [
                    {
                        "from": ["waiting"],
                        "event": "cue.brighten",
                        "action": {"kind": "enter", "stage": "bright"},
                    }
                ],
            }
        },
    }
    voice["bindings"] = [
        {"name": "clock", "kind": "motion", "reference": "clock"},
        {"name": "level", "kind": "motion", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["event_connections"] = [
        {
            "source": "clock",
            "port": "peak",
            "destination": "level",
            "cue": "brighten",
        }
    ]
    if cascade:
        voice["motions"]["relay"] = {
            "body": {
                "kind": "stages",
                "initial_stage": "waiting",
                "stages": [
                    {"name": "waiting", "motion": {"kind": "hold", "value": 0.0}}
                ],
                "transitions": [
                    {
                        "from": ["waiting"],
                        "event": "cue.fire",
                        "action": {"kind": "finish"},
                    }
                ],
            }
        }
        voice["bindings"].append(
            {"name": "relay", "kind": "motion", "reference": "relay"}
        )
        voice["modulation"]["sources"].append(
            {"name": "relay", "scope": "voice", "minimum": -1, "maximum": 1}
        )
        voice["event_connections"] = [
            {"source": "clock", "port": "peak", "destination": "relay", "cue": "fire"},
            {
                "source": "relay",
                "port": "done",
                "destination": "level",
                "cue": "brighten",
            },
        ]
    document = SynthInstrumentScore.model_validate(raw)
    clock = document.body.voices[0].motions["clock"]
    probe = advance_motion(
        clock,
        initial_motion(clock, Fraction(0)),
        Fraction(12001, 48000),
    )
    assert [e.port for e in probe.events] == ["peak"]
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(prepared, backend).advance(actions, 0, 48000)
    renderer = synth.OfflineSynth(prepared, backend)
    first = renderer.advance(actions, 0, 16000)
    snapshot = renderer.snapshot()
    assert snapshot.envelopes[1].state.runtime.stage == "bright"
    second = renderer.advance([], 16000, 48000)
    restored = synth.OfflineSynth(prepared, backend)
    restored.restore(snapshot)
    replay = restored.advance([], 16000, 48000)
    case = f"{cascade}-{position.replace('/', '-')}"
    check_audio(
        tmp_path / f"staged-connection-{backend}-{case}.wav",
        np.concatenate((first, second)),
        expected,
    )
    np.testing.assert_allclose(replay, second, atol=0)
    assert expected[12000, 0] == pytest.approx(1 if position == "1/4" else 0.625)
    assert expected[12001, 0] == pytest.approx(1)
    if backend == "native":
        persistent = synth.PersistentSynth(prepared, voices=4)
        native_first = persistent.advance(actions, 0, 16000)
        native_snapshot = persistent.snapshot()
        native_second = persistent.advance([], 16000, 48000)
        native_restored = synth.PersistentSynth(prepared, voices=4)
        native_restored.restore(native_snapshot)
        native_replay = native_restored.advance([], 16000, 48000)
        check_audio(
            tmp_path / f"persistent-staged-connection-{case}.wav",
            np.concatenate((native_first, native_second)),
            expected,
        )
        np.testing.assert_allclose(native_replay, native_second, atol=0)


@pytest.mark.parametrize("child_kind", ["cycle", "stages"])
@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_patch_named_event_output_cues_stage(
    tmp_path: Path, backend: Literal["numpy", "native"], child_kind: str
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    pulse = {
        "kind": "cycle",
        "rate": "1",
        "markers": [{"name": "peak", "position": "1/4"}],
    }
    if child_kind == "stages":
        pulse = {
            "kind": "stages",
            "initial_stage": "playing",
            "stages": [{"name": "playing", "motion": pulse}],
        }
    voice["motions"] = {
        "patch": {
            "body": {
                "kind": "patch",
                "motions": {"carrier": {"kind": "cycle", "rate": "1"}, "pulse": pulse},
                "outputs": {"signal": "carrier"},
                "event_outputs": {"strike": "pulse.peak"},
            }
        },
        "level": {
            "body": {
                "kind": "stages",
                "initial_stage": "waiting",
                "stages": [
                    {"name": "waiting", "motion": {"kind": "hold", "value": 0}},
                    {"name": "bright", "motion": {"kind": "hold", "value": 1}},
                ],
                "transitions": [
                    {
                        "from": ["waiting"],
                        "event": "cue.brighten",
                        "action": {"kind": "enter", "stage": "bright"},
                    }
                ],
            }
        },
    }
    voice["bindings"] = [
        {"name": "patch", "kind": "motion", "reference": "patch", "output": "signal"},
        {"name": "level", "kind": "motion", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "patch", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["event_connections"] = [
        {"source": "patch", "port": "strike", "destination": "level", "cue": "brighten"}
    ]
    document = SynthInstrumentScore.model_validate(raw)
    prepared = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(prepared, backend).advance(actions, 0, 48000)
    renderer = synth.OfflineSynth(prepared, backend)
    first = renderer.advance(actions, 0, 16000)
    snapshot = renderer.snapshot()
    second = renderer.advance([], 16000, 48000)
    restored = synth.OfflineSynth(prepared, backend)
    restored.restore(snapshot)
    np.testing.assert_allclose(restored.advance([], 16000, 48000), second, atol=0)
    check_audio(
        tmp_path / f"patch-event-{child_kind}-{backend}.wav",
        np.concatenate((first, second)),
        expected,
    )
    assert expected[12000, 0] == pytest.approx(1)
    if backend == "native":
        persistent = synth.PersistentSynth(prepared, voices=4)
        native_first = persistent.advance(actions, 0, 16000)
        native_snapshot = persistent.snapshot()
        native_second = persistent.advance([], 16000, 48000)
        native_restored = synth.PersistentSynth(prepared, voices=4)
        native_restored.restore(native_snapshot)
        np.testing.assert_allclose(
            native_restored.advance([], 16000, 48000), native_second, atol=0
        )
        check_audio(
            tmp_path / f"patch-event-{child_kind}-persistent.wav",
            np.concatenate((native_first, native_second)),
            expected,
        )


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
            "motions": {
                "motion": {
                    "scope": scope,
                    "body": {"kind": "cycle", "rate": "0", "phase": "3/4"},
                }
            },
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
