from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.events import Release
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("clock", ["seconds", "beats"])
def test_rational_named_loop_bounds_and_continuation_match_reference(
    tmp_path: Path, clock: str
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"]["envelfo"] = {
        "kind": "envelope",
        "clock": clock,
        "playback": "ping_pong",
        "repeat_count": 3,
        "segments": [
            {"duration": f"1/7 {'s' if clock == 'seconds' else 'beat'}", "to": 1}
        ],
        "markers": [
            {"name": "start", "position": "1/3"},
            {"name": "end", "position": "2/3"},
        ],
        "loop_start": "start",
        "loop_end": "end",
    }
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    definition = synth.prepare(document)
    tempo = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "137"},
                {
                    "at_seconds": "1/21",
                    "beat": "137/1260",
                    "bpm": "137",
                    "running": False,
                },
                {"at_seconds": "1/7", "beat": "8", "bpm": "91"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body, [onset(pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    expected = synth.OfflineSynth(definition, tempo_map=tempo).advance(
        actions, 0, 48000
    )
    renderer = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    first = renderer.advance(actions, 0, 4001)
    restored = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    restored.restore(renderer.snapshot())
    tail = renderer.advance([], 4001, 48000)
    replay = np.concatenate(
        [
            restored.advance([], 4001, 5000),
            restored.advance([], 5000, 17003),
            restored.advance([], 17003, 48000),
        ]
    )
    check_audio(
        tmp_path / f"rational-loop-{clock}.wav", np.concatenate([first, tail]), expected
    )
    np.testing.assert_allclose(replay, tail, atol=0)


@pytest.mark.parametrize("marker", ["1/4", "1/3"])
@pytest.mark.parametrize("source", ["stages", "lfo"])
@pytest.mark.parametrize("delay", ["0 s", "1/336000 s", "1/7 s"])
def test_fractional_routed_delivery_and_pending_queue_survive_restore(
    tmp_path: Path, marker: str, source: str, delay: str
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"] = {
        "gesture": {
            "kind": "patch",
            "envelfos": {
                "clock": {
                    "kind": "stages",
                    "initial_stage": "pulse",
                    "stages": [
                        {
                            "name": "pulse",
                            "envelfo": {
                                "kind": "envelope",
                                "segments": [{"duration": "1/48000 s", "to": 1}],
                                "markers": [{"name": "mark", "position": marker}],
                            },
                        }
                    ],
                },
                "accent": {
                    "kind": "envelope",
                    "start": "event",
                    "segments": [{"duration": "1 s", "to": 1}],
                },
            },
            "events": [
                {
                    "source": "clock.mark",
                    "target": "accent",
                    "action": "start",
                    "delay": delay,
                }
            ],
            "outputs": {"value": "accent"},
        }
    }
    if source == "lfo":
        voice["envelfos"]["gesture"]["envelfos"]["clock"] = {
            "kind": "lfo",
            "rate": "1",
            "markers": [{"name": "mark", "position": str(Fraction(marker) / 48000)}],
        }
    voice["bindings"] = [
        {"name": "level", "kind": "envelfo", "reference": "gesture", "output": "value"}
    ]
    voice["modulation"]["sources"] = [
        {"name": "level", "scope": "voice", "minimum": 0, "maximum": 1}
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    definition = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body, [onset(pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    expected = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    assert expected[9000, 0] == pytest.approx(
        float(
            (9000 - Fraction(marker) - Fraction(delay.removesuffix(" s")) * 48000)
            / 48000
        )
    )
    renderer = synth.PersistentSynth(definition, voices=2)
    first = renderer.advance(actions, 0, 1)
    restored = synth.PersistentSynth(definition, voices=2)
    restored.restore(renderer.snapshot())
    tail = renderer.advance([], 1, 48000)
    actual = np.concatenate([first, tail])
    np.testing.assert_allclose(restored.advance([], 1, 48000), tail, atol=0)
    check_audio(
        tmp_path / f"fractional-start-{marker.replace('/', '-')}.wav", actual, expected
    )


def test_fractional_minimum_hold_keeps_voice_timed_named_release_exact(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["minimum_hold_seconds"] = "1/7 s"
    voice["envelope"]["release"] = [{"duration": "1 s", "to": 0}]
    voice["envelfos"]["envelfo"] = {
        "kind": "envelope",
        "initial": 1,
        "segments": [{"duration": None}],
        "release": [{"duration": "1/3 s", "to": 0}],
    }
    voice["bindings"][0]["release_timing"] = "voice"
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    definition = synth.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=0.1).model_copy(update={"controls": {}}),
            Release(tick=1, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    expected = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    actual = synth.PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(tmp_path / "fractional-voice-release.wav", actual, expected)
