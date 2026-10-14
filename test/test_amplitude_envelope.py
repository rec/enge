from pathlib import Path

import numpy as np
import pytest
import test_fm
import test_noise
import test_synth
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from ufor import synth_trace
from ufor.events import EnvelfoChange, Release
from ufor.samples import instrument, trace
from ufor.synth import SynthInstrumentScore

from enge import fm, live, noise, sample_instrument, synth


def document_for(family: str) -> SynthInstrumentScore:
    document = {
        "synth": test_synth.score,
        "fm": test_fm.score,
        "noise": test_noise.score,
    }[family]()
    raw = document.model_dump(mode="json")
    envelope = {
        "segments": [
            {"duration": "1/8 s", "to": 1},
            {"duration": None},
            {"duration": "1/8 s", "to": 0.5},
        ],
        "release": [{"duration": None}, {"duration": "1/8 s", "to": 0}],
    }
    voice = raw["body"]["voices"][0]
    if family == "fm":
        for operator in voice["fm"]["operators"]:
            operator["envelope"] = envelope
    else:
        voice["envelope"] = envelope
    return SynthInstrumentScore.model_validate(raw)


def commands(op: str | None = None) -> list[EnvelfoChange]:
    return [
        EnvelfoChange(
            tick=t,
            ordinal=1,
            part="main",
            trigger_id="note",
            target="envelope",
            op=op,
            action="advance",
        )
        for t in (12000, 30000)
    ]


@pytest.mark.parametrize("family", ["synth", "fm", "noise"])
def test_voice_holds_release_and_restore_match_all_backends(
    tmp_path: Path, family: str
) -> None:
    document = document_for(family)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=220).model_copy(update={"controls": {}}),
            Release(tick=24000, ordinal=0, part="main", trigger_id="note"),
            *commands(),
        ],
        seed=0,
    ).actions
    if family == "fm":
        definition = fm.prepare(document)
        reference = fm.OfflineFM(definition).advance(actions, 0, 48000)
        offline = fm.OfflineFM(definition, "native")
        persistent = fm.PersistentFM(definition, voices=2)
        restored = fm.PersistentFM(definition, voices=2)
    elif family == "noise":
        definition = noise.prepare(document)
        reference = noise.OfflineNoise(definition).advance(actions, 0, 48000)
        offline = noise.OfflineNoise(definition, "native")
        persistent = noise.PersistentNoise(definition, voices=2)
        restored = noise.PersistentNoise(definition, voices=2)
    else:
        definition = synth.prepare(document)
        reference = synth.OfflineSynth(definition).advance(actions, 0, 48000)
        offline = synth.OfflineSynth(definition, "native")
        persistent = synth.PersistentSynth(definition, voices=2)
        restored = synth.PersistentSynth(definition, voices=2)
    test_synth.check_audio(
        tmp_path / f"{family}-offline.wav",
        offline.advance(actions, 0, 48000),
        reference,
    )
    prefix = persistent.advance([a for a in actions if a.tick < 28000], 0, 28000)
    restored.restore(persistent.snapshot())
    tail_actions = [a for a in actions if a.tick >= 28000]
    tail = persistent.advance(tail_actions, 28000, 48000)
    test_synth.check_audio(
        tmp_path / f"{family}-persistent.wav", np.concatenate([prefix, tail]), reference
    )
    np.testing.assert_allclose(
        restored.advance(tail_actions, 28000, 48000), tail, atol=0
    )
    assert np.any(reference[24000:30000])
    # Noise filters may retain a short tail after the amplitude finishes.
    assert np.all(reference[42000:] == 0)


def test_fm_default_and_explicit_carrier_share_playback(tmp_path: Path) -> None:
    document = document_for("fm")
    start = onset(pitch=220).model_copy(update={"controls": {}})
    release = Release(tick=24000, ordinal=0, part="main", trigger_id="note")
    default = synth_trace.prepare(
        document.body, [start, release, *commands()], seed=0
    ).actions
    explicit = synth_trace.prepare(
        document.body, [start, release, commands()[0], commands("carrier")[1]], seed=0
    ).actions
    for renderer in (
        fm.OfflineFM(fm.prepare(document)),
        fm.PersistentFM(fm.prepare(document), voices=2),
    ):
        first = renderer.advance(default, 0, 48000)
        other = type(renderer)(fm.prepare(document))
        test_synth.check_audio(
            tmp_path / f"{type(renderer).__name__}-carrier.wav",
            other.advance(explicit, 0, 48000),
            first,
        )


def test_fm_operator_advance_does_not_advance_carrier(tmp_path: Path) -> None:
    document = document_for("fm")
    start = onset(pitch=220).model_copy(update={"controls": {}})
    action = EnvelfoChange(
        tick=12000,
        ordinal=0,
        part="main",
        trigger_id="note",
        target="envelope",
        op="modulator",
        action="advance",
    )
    actions = synth_trace.prepare(document.body, [start, action], seed=0).actions
    definition = fm.prepare(document)
    expected = fm.OfflineFM(definition).advance(actions, 0, 48000)
    actual = fm.PersistentFM(definition, voices=2).advance(actions, 0, 48000)
    unchanged = fm.OfflineFM(definition).advance(
        synth_trace.prepare(document.body, [start], seed=0).actions, 0, 48000
    )
    test_synth.check_audio(tmp_path / "modulator.wav", actual, expected)
    np.testing.assert_allclose(actual[:12000], unchanged[:12000], atol=1e-10)
    assert not np.allclose(actual[18000:], unchanged[18000:])


def test_amplitude_command_targets_only_its_trigger(tmp_path: Path) -> None:
    document = document_for("synth")
    first = onset(pitch=220).model_copy(update={"controls": {}})
    other = onset(9000, "other", pitch=330).model_copy(update={"controls": {}})
    change = commands()[0]
    actions = synth_trace.prepare(document.body, [first, other, change], seed=0).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    actual = synth.PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    test_synth.check_audio(tmp_path / "trigger-address.wav", actual, expected)
    baseline = synth.OfflineSynth(definition).advance(
        synth_trace.prepare(document.body, [first, other], seed=0).actions, 0, 48000
    )
    changed = synth.OfflineSynth(definition).advance(
        synth_trace.prepare(document.body, [first, change], seed=0).actions, 0, 48000
    )
    unchanged = synth.OfflineSynth(definition).advance(
        synth_trace.prepare(document.body, [first], seed=0).actions, 0, 48000
    )
    np.testing.assert_allclose(actual - baseline, changed - unchanged, atol=1e-10)


def test_sample_holds_and_native_live_snapshot_resume(tmp_path: Path) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["settings"]["envelope"] = {
        "initial": 1,
        "segments": [{"duration": None}],
        "release": [{"duration": None}, {"duration": "1/8 s", "to": 0}],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.25, 0.125], (96000, 1))}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
            EnvelfoChange(
                tick=30000,
                ordinal=0,
                part="main",
                trigger_id="note",
                target="envelope",
                action="advance",
            ),
        ],
        seed=0,
    ).actions
    reference = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    test_synth.check_audio(
        tmp_path / "sample-native.wav",
        sample_instrument.OfflineSampler(definition, "native").advance(
            actions, 0, 48000
        ),
        reference,
    )
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition, voices=2)}, 28000
    )
    actual = np.empty_like(reference)
    engine.advance_into(
        {"sample": [a for a in actions if a.tick < 28000]}, [], 0, 28000, actual[:28000]
    )
    saved = engine.snapshot()
    remaining = {"sample": [a for a in actions if a.tick >= 28000]}
    engine.advance_into(remaining, [], 28000, 48000, actual[28000:])
    restored = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition, voices=2)}, 28000
    )
    restored.restore(saved)
    replay = np.empty((20000, 2))
    restored.advance_into(remaining, [], 28000, 48000, replay)
    test_synth.check_audio(tmp_path / "sample-live.wav", actual, reference)
    np.testing.assert_allclose(replay, actual[28000:], atol=0)
    assert np.all(reference[12000:30000, 0] == 0.25)
    np.testing.assert_allclose(
        reference[30000:36000, 0], 0.25 * (1 - np.arange(6000) / 6000)
    )
    assert np.all(reference[36000:] == 0)


def test_exhausted_sample_ends_despite_indefinite_release(tmp_path: Path) -> None:
    raw = sample_score(frames=18000).model_dump(mode="json")
    raw["body"]["settings"]["envelope"] = {
        "initial": 1,
        "segments": [{"duration": None}],
        "release": [{"duration": None}],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.25, 0.125], (18000, 1))}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
            EnvelfoChange(
                tick=30000,
                ordinal=0,
                part="main",
                trigger_id="note",
                target="envelope",
                action="advance",
            ),
        ],
        seed=0,
    ).actions
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition, voices=2)}, 48000
    )
    actual = np.empty_like(expected)
    engine.advance_into({"sample": actions}, [], 0, 48000, actual)
    test_synth.check_audio(tmp_path / "exhausted-sample.wav", actual, expected)
    assert np.all(actual[:18000, 0] == 0.25)
    assert np.all(actual[18000:] == 0)
