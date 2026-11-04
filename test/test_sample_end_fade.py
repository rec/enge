from math import ceil
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import change, dynamic_score, onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import Release
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize(
    ("direction", "repetition"),
    [
        (d, r)
        for d in ("forward", "backward", "mirror")
        for r in ("none", "whole", "loop", "overlap")
        if (d, r) != ("mirror", "overlap")
    ],
)
@pytest.mark.parametrize("step", [0.5, 1, 7])
def test_end_fade_runs_once_before_final_exhaustion(
    tmp_path: Path, direction: str, repetition: str, step: float
) -> None:
    raw = sample_score(frames=12).model_dump(mode="json")
    playback = raw["body"]["slots"][0]["playback"]
    playback.update(direction=direction, end_fade_seconds=4 / 48000)
    length = 23 if direction == "mirror" else 12
    if repetition == "whole":
        playback.update(mode="one_shot", play_count=3)
        length *= 3
    elif repetition in ("loop", "overlap"):
        overlap = 2 if repetition == "overlap" else 0
        raw["body"]["slices"][0]["loop"] = {
            "start_frame": 3,
            "end_frame": 9,
            "crossfade_frames": overlap,
            "repeat_count": 3,
            "mode": "through_release",
        }
        length = 12 + 3 * (5 if direction == "mirror" else 6 - overlap)
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.2, 0.1], (12, 1))}
    )
    actions = trace.prepare(
        document.body,
        [onset(pitch=440 * step).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    end = ceil(length / step)
    start = max(0, ceil(length / step - 4))
    gain = np.zeros(48000)
    gain[:end] = np.minimum(1, np.maximum(0, 1 - (np.arange(end) - start) / 4))
    check_renderers(
        tmp_path, definition, actions, gain[:, None] * [0.2, 0.15], start + 1
    )


@pytest.mark.parametrize("initial_bend", [0, 1])
def test_pitch_changes_do_not_retime_started_end_fade(
    tmp_path: Path, initial_bend: int
) -> None:
    raw = sample_score(frames=12000).model_dump(mode="json")
    raw["body"]["slots"][0]["playback"]["end_fade_seconds"] = 0.125
    dynamic = dynamic_score(smoothing="0").model_dump(mode="json")["body"]
    raw["body"]["settings"].update(
        controls=dynamic["controls"],
        modulation=dynamic["voices"][0]["modulation"],
        bindings=dynamic["voices"][0]["bindings"],
    )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.2, 0.1], (12000, 1))}
    )
    change_at = 3000 if initial_bend else 9000
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440, gain=1).model_copy(
                update={"controls": {"gain": 1, "bend": initial_bend}}
            ),
            change(change_at, 1 - initial_bend, control="bend"),
        ],
        seed=0,
    ).actions
    fade_at = 0 if initial_bend else 6000
    gain = np.minimum(1, np.maximum(0, 1 - (np.arange(48000) - fade_at) / 6000))
    if not initial_bend:
        gain[10500:] = 0  # Faster traversal exhausts before the fade completes.
    check_renderers(
        tmp_path, definition, actions, gain[:, None] * [0.2, 0.15], change_at + 1001
    )


def test_released_infinite_loop_begins_end_fade_on_exit(tmp_path: Path) -> None:
    raw = sample_score(frames=12000).model_dump(mode="json")
    raw["body"]["slices"][0]["loop"] = {
        "start_frame": 3000,
        "end_frame": 9000,
        "mode": "until_release",
    }
    raw["body"]["slots"][0]["playback"]["end_fade_seconds"] = 0.125
    raw["body"]["settings"]["envelope"] = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [{"duration": 1, "to": 1}, {"duration": 0, "to": 0}],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.2, 0.1], (12000, 1))}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    gain = np.minimum(1, np.maximum(0, 1 - (np.arange(48000) - 18000) / 6000))
    check_renderers(tmp_path, definition, actions, gain[:, None] * [0.2, 0.15], 19001)


@pytest.mark.parametrize("order", ["before_amplitude", "after_amplitude"])
def test_end_fade_follows_filters_and_multiplies_retirement_fade(
    tmp_path: Path, order: str
) -> None:
    raw = sample_score(frames=12000).model_dump(mode="json")
    slot = raw["body"]["slots"][0]
    slot["playback"].update(start_delay_seconds=0.125, end_fade_seconds=0.125)
    raw["body"]["settings"]["processing"] = {
        "filter_order": order,
        "filters": [{"name": "tone", "response": "lowpass", "cutoff_hz": 100}],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.tile([0.2, 0.1], (12000, 1))
    definition = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body,
        [onset(pitch=440).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    voice = next(a.voice_id for a in actions if isinstance(a, trace.VoiceStart))
    actions.append(
        VoiceRetirement(
            tick=13000,
            ordinal=0,
            voice_id=voice,
            cause=RetirementCause.choke,
            action="fade",
            fade_seconds=0.125,
        )
    )
    slot["playback"]["end_fade_seconds"] = 0
    baseline = sample_instrument.prepare(
        instrument.SampleInstrumentScore.model_validate(raw), {"asset": audio}
    )
    expected = sample_instrument.OfflineSampler(baseline).advance(actions, 0, 48000)
    expected *= np.minimum(1, np.maximum(0, 1 - (np.arange(48000) - 12000) / 6000))[
        :, None
    ]
    check_renderers(tmp_path, definition, actions, expected, 14001)


def check_renderers(
    tmp_path: Path,
    definition: sample_instrument.PreparedSampler,
    actions: list[trace.Action],
    expected: np.ndarray,
    split: int,
) -> None:
    first_actions = [a for a in actions if a.tick < split]
    tail_actions = [a for a in actions if a.tick >= split]
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(first_actions, 0, split)
        saved = renderer.snapshot()
        saved = type(saved).model_validate_json(saved.model_dump_json())
        tail = renderer.advance(tail_actions, split, 48000)
        check_audio(
            tmp_path / f"end-fade-{backend}.wav",
            np.concatenate([first, tail]),
            expected,
        )
        assert renderer.snapshot().voices == []
        renderer.restore(saved)
        np.testing.assert_array_equal(
            renderer.advance(tail_actions, split, 48000), tail
        )
    source = sample_instrument.PersistentSampler(definition)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty_like(expected)
    engine.advance_into({"sample": first_actions}, [], 0, split, actual[:split])
    saved = engine.snapshot()
    engine.advance_into({"sample": tail_actions}, [], split, 48000, actual[split:])
    check_audio(tmp_path / "end-fade-live.wav", actual, expected)
    engine.restore(saved)
    replay = np.empty_like(actual[split:])
    engine.advance_into({"sample": tail_actions}, [], split, 48000, replay)
    np.testing.assert_array_equal(replay, actual[split:])
