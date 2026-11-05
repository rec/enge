"""Compare the native register captures with enge at host and chip cadence."""

import csv
import sys
import wave
from itertools import product
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from tyro import cli
from ufor import tx81z

from enge.yamaha import TX81ZRenderer, TX81ZState, tx81z_dac, tx81z_operator_samples


class Comparison(BaseModel, frozen=True):
    source: Path

    def run(self) -> None:
        table = csv.writer(sys.stdout, lineterminator="\n")
        table.writerow(
            [
                "algorithm",
                "feedback",
                "profile",
                "reference_peak",
                "host_relative_rms",
                "host_max_units",
                "native_mismatches",
                "native_max_units",
            ]
        )
        ticks = np.arange(1, 48001) * 3579545 // (64 * 48000)
        for algorithm, feedback, profile in product(range(1, 9), range(8), range(2)):
            stem = self.source / f"note-{algorithm}-{feedback}-{profile}"
            expected = np.fromfile(stem.with_suffix(".i32"), dtype="=i4")
            if (
                expected.shape != (48000,)
                or not np.any(expected[:24000])
                or not np.any(expected[24000:])
            ):
                raise ValueError(f"Invalid one-second note capture in {stem}")
            note = prepare_note(algorithm, feedback, profile)
            head = note.render(24000)
            note.release()
            raw = np.concatenate([head, note.render(24000)])
            host = tx81z_dac(raw[:, 0]) * 32768
            native = native_control(prepare_note(algorithm, feedback, profile), ticks)
            relative_rms = np.linalg.norm(host - expected) / np.linalg.norm(expected)
            table.writerow(
                [
                    algorithm,
                    feedback,
                    profile,
                    int(np.max(np.abs(expected))),
                    f"{relative_rms:.12f}",
                    int(np.max(np.abs(host - expected))),
                    np.count_nonzero(native != expected),
                    int(np.max(np.abs(native - expected))),
                ]
            )
            if profile == 0 and (algorithm, feedback) in ((8, 0), (1, 7)):
                for name, samples in (("ymfm", expected), ("enge", host)):
                    with wave.open(
                        str(stem.with_name(f"note-{algorithm}-{feedback}-{name}.wav")),
                        "wb",
                    ) as output:
                        output.setparams((1, 2, 48000, 48000, "NONE", ""))
                        output.writeframes(samples.astype("<i2").tobytes())


def prepare_note(algorithm: int, feedback: int, profile: int) -> TX81ZRenderer:
    raw, extra = bytearray(93), bytearray(23)
    for i in range(4):
        raw[i * 13 : i * 13 + 5] = bytes(
            [20, 16, 0, 15, 10] if profile else [31, 12, 4, 9, 7]
        )
        raw[i * 13 + 10 : i * 13 + 13] = bytes(
            [99, 4 + i if profile else 4, [0, 2, 4, 6][i] if profile else 3]
        )
        if profile:
            extra[i * 5 + 2 : i * 5 + 5] = bytes(
                [[0, 3, 8, 15][i], [0, 1, 4, 7][i], [0, 2, 1, 0][i]]
            )
    raw[52], raw[53], raw[62] = algorithm - 1, feedback, 24
    return TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(extra)),
        69,
        127,
    )


def native_control(note: TX81ZRenderer, ticks: np.ndarray) -> np.ndarray:
    """Drive production components at chip cadence using prepared native steps."""
    envelopes = note.envelopes.envelopes
    gains = np.empty((ticks[-1], 4))
    for k in range(1, ticks[-1] + 1):
        if k == 3579545 // 128 + 1:
            note.release()
        if k % 3 == 0:
            for envelope in envelopes:
                envelope.clock(k // 3)
        gains[k - 1] = [
            2 ** (-(e.attenuation >> e.shift) / 64) * g if e.attenuation <= 896 else 0
            for e, g in zip(envelopes, note.levels, strict=True)
        ]
    steps = np.rint(note.frequencies * (64 * 2**20 / 3579545)).astype(int)
    # The chip clocks before output. Start the production output-first kernel
    # at the first clocked phase and give it exactly one native step per row.
    frequencies = np.broadcast_to(steps * (48000 / 2**20), gains.shape)
    chip_rows, _ = tx81z_operator_samples(
        note.voice.algorithm,
        note.waveforms,
        TX81ZState(phases=steps.tolist()),
        frequencies,
        gains,
        feedback=note.voice.feedback,
    )
    return tx81z_dac(chip_rows[ticks - 1, 0]) * 32768


if __name__ == "__main__":
    cli(Comparison).run()
