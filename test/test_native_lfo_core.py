from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.events import EnvelfoChange, LFOChange
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("clock", ["seconds", "beats"])
@pytest.mark.parametrize("shape", ["square", "triangle"])
def test_decimal_commands_and_paused_reset_preserve_core_continuation(
    tmp_path: Path, clock: str, shape: str
) -> None:
    raw = lfo_score("synth", "instrument").model_dump(mode="json")
    raw["body"]["voices"][0]["envelfos"]["envelfo"].update(
        clock=clock,
        shape=shape,
        rate="7/3",
        phase="1/7",
        duty_cycle="1/3",
        delay="1/11",
        fade_in="1/7",
    )
    document = SynthInstrumentScore.model_validate(raw)
    tempo = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "137"},
                {
                    "at_seconds": "1/4",
                    "beat": "137/240",
                    "bpm": "137",
                    "running": False,
                },
                {"at_seconds": "1/2", "beat": "-2", "bpm": "91"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=0.1).model_copy(update={"controls": {}}),
            LFOChange(
                tick=1000, ordinal=0, name="envelfo", action="seek", position=0.1
            ),
            LFOChange(tick=5000, ordinal=0, name="envelfo", action="rate", rate=0.1),
            LFOChange(tick=9000, ordinal=0, name="envelfo", action="pause"),
            LFOChange(tick=11000, ordinal=0, name="envelfo", action="reset"),
            LFOChange(tick=16000, ordinal=0, name="envelfo", action="resume"),
            LFOChange(tick=20000, ordinal=1, name="envelfo", action="reverse"),
            LFOChange(
                tick=24000, ordinal=2, name="envelfo", action="shift", offset=-0.2
            ),
            LFOChange(tick=27000, ordinal=0, name="envelfo", action="rate", rate=0),
            LFOChange(tick=30000, ordinal=0, name="envelfo", action="rate", rate=7.3),
            LFOChange(
                tick=30000, ordinal=1, name="envelfo", action="seek", position=0.3
            ),
        ],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition, tempo_map=tempo).advance(
        actions, 0, 48000
    )
    renderer = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    prefix = renderer.advance([a for a in actions if a.tick < 13000], 0, 13000)
    restored = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    restored.restore(renderer.snapshot())
    remaining = [a for a in actions if a.tick >= 13000]
    tail = renderer.advance(remaining, 13000, 48000)
    replay = np.concatenate(
        [
            restored.advance([a for a in remaining if a.tick < 30000], 13000, 30000),
            restored.advance([a for a in remaining if a.tick >= 30000], 30000, 48000),
        ]
    )
    check_audio(
        tmp_path / f"commands-{clock}-{shape}.wav",
        np.concatenate([prefix, tail]),
        expected,
    )
    np.testing.assert_allclose(replay, tail, atol=0)


def test_transport_lfo_uses_exact_large_and_negative_song_beats(tmp_path: Path) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    raw["body"]["voices"][0]["envelfos"]["envelfo"] = {
        "kind": "lfo",
        "shape": "square",
        "rate": "1/3",
        "phase": "1/7",
        "duty_cycle": "1/3",
        "reset": "transport",
        "clock": "beats",
        "position_driver": "transport",
    }
    document = SynthInstrumentScore.model_validate(raw)
    tempo = TempoMap.model_validate(
        {
            "points": [
                {
                    "at_seconds": "0",
                    "beat": "9007199254740993",
                    "bpm": "137",
                    "running": False,
                },
                {"at_seconds": "1/2", "beat": "-2", "bpm": "137", "running": False},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body, [onset(pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    definition = synth.prepare(document)
    renderer = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    prefix = renderer.advance(actions, 0, 24000)
    restored = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    restored.restore(renderer.snapshot())
    tail = renderer.advance([], 24000, 48000)
    expected = np.zeros((48000, 2))
    # The large integer is divisible by three; the later phase is 10/21.
    expected[:24000, 0] = 1
    expected[24000:, 0] = 0.25
    check_audio(
        tmp_path / "transport-exact.wav", np.concatenate([prefix, tail]), expected
    )
    np.testing.assert_array_equal(restored.advance([], 24000, 48000), tail)


def test_voice_lfo_pause_retimes_routed_marker_and_restores_cursor(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"] = {
        "clock": {
            "kind": "lfo",
            "rate": "1",
            "markers": [{"name": "mark", "position": "1/3"}],
        },
        "level": {
            "kind": "stages",
            "initial_stage": "waiting",
            "stages": [
                {
                    "name": "waiting",
                    "envelfo": {
                        "kind": "envelope",
                        "initial": 1,
                        "segments": [{"duration": None}],
                    },
                },
                {
                    "name": "quiet",
                    "envelfo": {
                        "kind": "envelope",
                        "polarity": "bipolar",
                        "initial": -1,
                        "segments": [{"duration": None}],
                    },
                },
            ],
            "transitions": [
                {
                    "from": ["waiting"],
                    "event": "cue.quiet",
                    "action": {"kind": "enter", "stage": "quiet"},
                }
            ],
        },
    }
    voice["bindings"] = [
        {"name": "clock", "kind": "envelfo", "reference": "clock"},
        {"name": "level", "kind": "envelfo", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0].update(
        source="level", points=[{"input": -1, "amount": 0}, {"input": 1, "amount": 1}]
    )
    voice["event_connections"] = [
        {"source": "clock", "port": "mark", "destination": "level", "cue": "quiet"}
    ]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=0.1).model_copy(update={"controls": {}}),
            EnvelfoChange(
                tick=4800,
                ordinal=0,
                name="clock",
                part="main",
                trigger_id="note",
                action="pause",
            ),
            EnvelfoChange(
                tick=24000,
                ordinal=0,
                name="clock",
                part="main",
                trigger_id="note",
                action="resume",
            ),
        ],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    renderer = synth.PersistentSynth(definition, voices=2)
    prefix = renderer.advance([a for a in actions if a.tick < 20000], 0, 20000)
    restored = synth.PersistentSynth(definition, voices=2)
    restored.restore(renderer.snapshot())
    remaining = [a for a in actions if a.tick >= 20000]
    tail = renderer.advance(remaining, 20000, 48000)
    expected = np.zeros((48000, 2))
    expected[:35200, 0] = 1
    check_audio(
        tmp_path / "routed-marker.wav", np.concatenate([prefix, tail]), expected
    )
    np.testing.assert_array_equal(restored.advance(remaining, 20000, 48000), tail)
