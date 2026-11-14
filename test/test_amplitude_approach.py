from itertools import pairwise
from math import log
from pathlib import Path

import numpy as np
import pytest
import test_fm
import test_noise
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import trace

from enge import fm, live, noise, sample_instrument, synth


@pytest.mark.parametrize("kind", ["synth", "sampler"])
@pytest.mark.parametrize(
    "gate,seconds,start",
    [
        (0.05, 0.1, 0.2),
        (0.25, 0.1, 0.2),
        (0.55, 0.1, 0.2),
        (0.8, 0.1, 0.2),
        (0.8, 0.001, 0.2),
        (0.25, 0, 0.2),
        (0.05, 0.1, 0.00001),
        (0.05, 0.1, 0),
    ],
)
def test_amplitude_release_captures_level_and_survives_restore(
    tmp_path: Path,
    kind: str,
    gate: float,
    seconds: float,
    start: float,
) -> None:
    prototype = lfo_score(kind)
    raw = prototype.model_dump(mode="json")
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    voice.update(envelfos={}, bindings=[], modulation={})
    voice["envelope"] = {
        "initial": start,
        "segments": [
            {"duration": "0.1 s", "to": start},
            {"duration": "0.4 s", "to": 1},
            {"duration": "0.1 s", "to": 0.3, "curve": -5},
        ],
        "release": [
            {
                "kind": "exponential_approach",
                "asymptote": 0,
                "time_constant": seconds / 9,
                "minimum_time_constant": 0.006 / 9,
                "threshold": 0.0001,
                "transition_to": 0,
                "transition_seconds": 0.05,
                "minimum_transition_distance": 0.0001,
            }
        ],
    }
    document = type(prototype).model_validate(raw)
    voice["envelope"] = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [{"duration": "2 s", "to": 1}],
    }
    plain = type(prototype).model_validate(raw)
    release_frame = round(gate * 48000)
    events = [
        onset(pitch=440).model_copy(update={"controls": {}}),
        Release(tick=release_frame, ordinal=1, part="main", trigger_id="note"),
    ]
    if kind == "synth":
        prepared, baseline = synth.prepare(document), synth.prepare(plain)
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        base_actions = synth_trace.prepare(plain.body, events, seed=0).actions
        source = synth.OfflineSynth(baseline).advance(base_actions, 0, 96000)
    else:
        samples = {"asset": np.tile([0.125, 0.0625], (96000, 1))}
        prepared = sample_instrument.prepare(document, samples)
        baseline = sample_instrument.prepare(plain, samples)
        actions = trace.prepare(document.body, events, seed=0).actions
        base_actions = trace.prepare(plain.body, events, seed=0).actions
        source = sample_instrument.OfflineSampler(baseline).advance(
            base_actions, 0, 96000
        )
    expected = source * gains(np.arange(96000) / 48000, gate, seconds, start)[:, None]
    for backend in ["numpy", "native"]:
        renderer = (
            synth.OfflineSynth(prepared, backend)
            if kind == "synth"
            else sample_instrument.OfflineSampler(prepared, backend)
        )
        chunks = []
        for a, b in pairwise(
            sorted({0, release_frame, release_frame + 7, 48001, 96000})
        ):
            chunks.append(
                renderer.advance([v for v in actions if a <= v.tick < b], a, b)
            )
            snapshot = renderer.snapshot()
            restored = type(snapshot).model_validate_json(snapshot.model_dump_json())
            renderer = type(renderer)(prepared, backend)
            renderer.restore(restored)
        check_audio(
            tmp_path / f"{kind}-{backend}.wav", np.concatenate(chunks), expected
        )
    persistent = (
        synth.PersistentSynth(prepared, voices=1)
        if kind == "synth"
        else sample_instrument.PersistentSampler(prepared, voices=1)
    )
    split = release_frame + 7
    head = persistent.advance([v for v in actions if v.tick < split], 0, split)
    saved = persistent.snapshot()
    tail = persistent.advance([], split, 96000)
    persistent.restore(saved)
    replay = np.concatenate(
        [persistent.advance([], split, 48001), persistent.advance([], 48001, 96000)]
    )
    check_audio(tmp_path / f"{kind}-continuation.wav", replay, tail)
    check_audio(
        tmp_path / f"{kind}-persistent.wav", np.concatenate([head, tail]), expected
    )
    callback = live.NativeLiveEngine(
        {
            kind: synth.PersistentSynth(prepared, voices=1)
            if kind == "synth"
            else sample_instrument.PersistentSampler(prepared, voices=1)
        },
        maximum_block_frames=48000,
    )
    actual = np.empty_like(expected)
    for a, b in pairwise(sorted({0, release_frame, split, 48001, 96000})):
        callback.advance_into(
            {kind: [v for v in actions if a <= v.tick < b]}, [], a, b, actual[a:b]
        )
        if b == split:
            callback_saved = callback.snapshot()
    callback.restore(callback_saved)
    replay = np.empty((96000 - split, 2))
    for a, b in [(split, 48001), (48001, 96000)]:
        callback.advance_into({kind: []}, [], a, b, replay[a - split : b - split])
    check_audio(tmp_path / f"{kind}-callback.wav", actual, expected)
    check_audio(tmp_path / f"{kind}-callback-restore.wav", replay, actual[split:])


@pytest.mark.parametrize("kind", ["noise", "fm"])
def test_generated_source_release_uses_the_shared_amplitude_lifetime(
    tmp_path: Path,
    kind: str,
) -> None:
    prototype = test_fm.score() if kind == "fm" else test_noise.score()
    raw = prototype.model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    held = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [{"duration": "2 s", "to": 1}],
    }
    if kind == "fm":
        voice.update(envelfos={}, bindings=[], modulation={}, processing={})
        for operator in voice["fm"]["operators"]:
            operator["envelope"] = held
        for edge in voice["fm"]["edges"]:
            edge["index"] = 0
        owner = next(
            o for o in voice["fm"]["operators"] if o["name"] == voice["fm"]["carrier"]
        )
    else:
        owner = voice
        owner["envelope"] = held
    plain = type(prototype).model_validate(raw)
    owner["envelope"] = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [
            {
                "kind": "exponential_approach",
                "asymptote": 0,
                "time_constant": 0.1 / 9,
                "minimum_time_constant": 0.006 / 9,
                "threshold": 0.0001,
                "transition_to": 0,
                "transition_seconds": 0.05,
                "minimum_transition_distance": 0.0001,
            }
        ],
    }
    document = type(prototype).model_validate(raw)
    events = [
        onset(pitch=440).model_copy(update={"controls": {}}),
        Release(tick=12000, ordinal=1, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    base_actions = synth_trace.prepare(plain.body, events, seed=0).actions
    if kind == "fm":
        prepared, baseline = fm.prepare(document), fm.prepare(plain)
        source = fm.OfflineFM(baseline).advance(base_actions, 0, 96000)
        renderers = [fm.OfflineFM(prepared, b) for b in ["numpy", "native"]]
        persistent = fm.PersistentFM(prepared, voices=1)
    else:
        prepared, baseline = noise.prepare(document), noise.prepare(plain)
        source = noise.OfflineNoise(baseline).advance(base_actions, 0, 96000)
        renderers = [noise.OfflineNoise(prepared, b) for b in ["numpy", "native"]]
        persistent = noise.PersistentNoise(prepared, voices=1)
    # Release precedes decay, so this onset remains constant at one.
    expected = source * gains(np.arange(96000) / 48000, 0.25, 0.1, 1)[:, None]
    for index, renderer in enumerate(renderers):
        check_audio(
            tmp_path / f"{kind}-{index}.wav",
            renderer.advance(actions, 0, 96000),
            expected,
        )
    callback = live.NativeLiveEngine({kind: persistent}, maximum_block_frames=48000)
    actual = np.empty_like(expected)
    callback.advance_into({kind: actions}, [], 0, 48000, actual[:48000])
    callback.advance_into({kind: []}, [], 48000, 96000, actual[48000:])
    check_audio(tmp_path / f"{kind}-callback.wav", actual, expected)


def gains(at: np.ndarray, gate: float, seconds: float, start: float) -> np.ndarray:
    def onset_level(age: np.ndarray) -> np.ndarray:
        attack = start + (1 - start) * np.clip((age - 0.1) / 0.4, 0, 1)
        decay = 1 - 0.7 * np.expm1(-5 * np.clip((age - 0.5) / 0.1, 0, 1)) / np.expm1(-5)
        return np.where(age < 0.1, start, np.where(age < 0.5, attack, decay))

    initial = float(onset_level(np.asarray(gate)))
    duration = (
        round(max(seconds, 0.006) / 9 * log(initial / 0.0001) * 1e9) / 1e9
        if seconds and initial > 0.0001
        else 0
    )
    terminal = 0.0001 if duration else initial
    fade = round(0.05 * min(1, terminal / 0.0001) * 1e9) / 1e9
    age = np.maximum(0, at - gate)
    exponential = (
        initial * np.exp(log(0.0001 / initial) * np.clip(age / duration, 0, 1))
        if duration
        else np.full(at.shape, initial)
    )
    fading = (
        terminal * np.clip(1 - (age - duration) / fade, 0, 1)
        if fade
        else np.zeros(at.shape)
    )
    return np.where(
        at < gate, onset_level(at), np.where(age < duration, exponential, fading)
    )
