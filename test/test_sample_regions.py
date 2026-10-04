from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError
from test_synth import check_audio
from ufor.arpeggiator_capture import CapturedPhrase, SourceNote
from ufor.samples.playback import Loop, Slice
from ufor.time import Timebase

from enge.sample_regions import RegionPlayer, render_regions


def _phrase() -> CapturedPhrase:
    boundaries = [0, 12_000, 32_000, 48_000]
    return CapturedPhrase(
        capture_id="spoken",
        timebase=Timebase.model_validate(
            {"name": "source-frames", "rate": {"numerator": 48_000}}
        ),
        end_tick=48_000,
        notes=[
            SourceNote(
                capture_id="spoken",
                note_id=name,
                onset_tick=start,
                gate_end_tick=27_000 if name == "b" else end,
                cell_end_tick=end,
                selection_key=key,
                region=Slice(
                    name=name,
                    asset="voice",
                    start_frame=start,
                    end_frame=end,
                ),
            )
            for name, key, start, end in zip(
                "abc", [60, 62, 64], boundaries[:-1], boundaries[1:], strict=True
            )
        ],
    )


def test_source_order_reconstructs_every_decoded_frame(
    backend: Literal["numpy", "native"], tmp_path: Path
) -> None:
    phrase = _phrase()
    source = np.column_stack(
        [np.arange(48_000, dtype=np.float64) / 48_000, np.arange(48_000) / -48_000]
    )
    actual = render_regions(
        phrase, phrase.notes, {"voice": source}, [1, 0], backend=backend
    )
    expected = source[:, [1, 0]]
    np.testing.assert_array_equal(actual, expected)
    check_audio(tmp_path / "source-order.wav", actual, expected)


def test_reordered_and_repeated_regions_follow_selected_identity(
    backend: Literal["numpy", "native"], tmp_path: Path
) -> None:
    phrase = _phrase()
    source = np.arange(48_000, dtype=np.float64)[:, None] / 48_000
    notes = [phrase.notes[i] for i in [2, 0, 0, 1]]
    actual = render_regions(phrase, notes, {"voice": source}, [0], backend=backend)
    expected = np.concatenate(
        [source[32_000:], source[:12_000], source[:12_000], source[12_000:32_000]]
    )
    np.testing.assert_array_equal(actual, expected)
    check_audio(tmp_path / "reordered.wav", actual, expected)


def test_seam_fade_applies_only_when_source_regions_jump(
    backend: Literal["numpy", "native"], tmp_path: Path
) -> None:
    phrase = _phrase()
    source = np.ones((48_000, 1), dtype=np.float64)
    source[12_000:32_000] = -1
    source[32_000:] = 0.5
    identity = render_regions(
        phrase,
        phrase.notes,
        {"voice": source},
        [0],
        backend=backend,
        seam_fade_frames=8,
    )
    np.testing.assert_array_equal(identity, source)
    notes = [phrase.notes[i] for i in [2, 0, 1]]
    actual = render_regions(
        phrase,
        notes,
        {"voice": source},
        [0],
        backend=backend,
        seam_fade_frames=8,
    )
    expected = np.concatenate([source[32_000:], source[:32_000]]).copy()
    fade = np.arange(1, 9, dtype=np.float64) / 8
    expected[16_000 - 8 : 16_000] *= (1 - fade)[:, None]
    expected[16_000 : 16_000 + 8] *= (fade - 1 / 8)[:, None]
    np.testing.assert_array_equal(actual, expected)
    check_audio(tmp_path / "seam-fade.wav", actual, expected)

    player = RegionPlayer(
        phrase=phrase,
        notes=notes,
        audio={"voice": source},
        channels=[0],
        backend=backend,
        seam_fade_frames=8,
    )
    chunks = [player.advance(n) for n in [127, 16_000, 997, 30_876]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)


def test_gate_policy_crops_the_tail_and_streams_in_irregular_blocks(
    backend: Literal["numpy", "native"], tmp_path: Path
) -> None:
    phrase = _phrase()
    source = np.arange(48_000, dtype=np.float64)[:, None] / 48_000
    player = RegionPlayer(
        phrase=phrase,
        notes=phrase.notes,
        audio={"voice": source},
        channels=[0],
        backend=backend,
        tail_policy="gate",
    )
    assert player.total_frames == 43_000
    blocks = [player.advance(n) for n in [127, 12_001, 997, 34_875]]
    actual = np.concatenate(blocks)
    expected = np.concatenate(
        [source[:12_000], source[12_000:27_000], source[32_000:], np.zeros((5_000, 1))]
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        render_regions(
            phrase,
            phrase.notes,
            {"voice": source},
            [0],
            backend=backend,
            tail_policy="gate",
        ),
        expected[:43_000],
    )
    check_audio(tmp_path / "gate-tail.wav", actual, expected)
    player.stop()
    np.testing.assert_array_equal(player.advance(128), np.zeros((128, 1)))


def test_looped_region_requires_an_explicit_playback_policy() -> None:
    phrase = _phrase()
    note = phrase.notes[0]
    looped = note.model_copy(
        update={
            "region": Slice(
                name="a",
                asset="voice",
                start_frame=0,
                end_frame=12_000,
                loop=Loop(start_frame=1, end_frame=11_999),
            )
        }
    )
    phrase = phrase.model_copy(update={"notes": [looped, *phrase.notes[1:]]})
    with pytest.raises(ValidationError, match="playback policy"):
        render_regions(phrase, [looped], {}, [0])
