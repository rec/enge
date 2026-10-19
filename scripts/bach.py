"""Render Bach's Little Fugue with FM bass/soprano, synth tenor, sampled alto."""

from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel, Field
from ufor import performance_binding
from ufor.interface import ScoreReference

from enge.midi import read_midi
from enge.performance import MidiConverter
from enge.presets import Patch
from enge.render import render_midi


class Options(BaseModel, frozen=True):
    output: Path = Path("bwv-578.flac")
    block_size: int = Field(default=4096, gt=0)
    backend: Literal["numpy", "native"] = "numpy"
    control_interval: int = Field(default=1, strict=True, gt=0)


def main() -> None:
    options = tyro.cli(Options)
    patches = {
        "bass": Patch(engine="fm", gain=0.24, pan=-0.2, ratio=1, index=0.65),
        "tenor": Patch(engine="synth", gain=0.20, pan=-0.55),
        "alto": Patch(engine="sample", gain=0.23, pan=0.45),
        "soprano": Patch(engine="fm", gain=0.20, pan=0.15, ratio=2, index=0.85),
    }
    instruments = {n: p.prepare_score(48000) for n, p in patches.items()}
    converters = []
    for channel, (part, (score, _)) in enumerate(instruments.items(), start=1):
        # The host resolves these selectors from its in-memory instruments map.
        binding = performance_binding.PerformanceBindingScore(
            name=f"bach-{part}",
            title=f"Bach {part}",
            body=performance_binding.PerformanceInputBinding(
                instrument=ScoreReference(selector=f"bach:{part}"),
                midi=[
                    performance_binding.MidiPerformanceInput(
                        channels=[channel],
                        part=part,
                        repeated_key_release="oldest",
                        controllers=[
                            performance_binding.MidiController(number=c, control=n)
                            for c, n in [(1, "modulation"), (11, "expression")]
                        ],
                        pitch_bend=performance_binding.MidiPitchBend(control="bend"),
                    )
                ],
            ),
        )
        converters.append(
            MidiConverter(source=f"bach-{part}", binding=binding, instrument=score)
        )
    performance = read_midi(
        Path(__file__).with_name("bwv-578.mid"), converters=converters
    )
    report = render_midi(
        performance,
        instruments,
        options.output,
        options.block_size,
        options.backend,
        progress=True,
        control_interval=options.control_interval,
    )
    print(
        f"{options.output}: {report.frames / report.sample_rate:.2f}s, "
        f"{report.notes} notes, peak {report.peak:.4f}; "
        f"rendered in {report.elapsed_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
