"""Render Bach's Little Fugue with FM bass/soprano, synth tenor, sampled alto."""

from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel, Field

from enge.midi import read_midi
from enge.presets import Patch
from enge.render import render_midi


class Options(BaseModel, frozen=True):
    output: Path = Path("bwv-578.flac")
    block_size: int = Field(default=4096, gt=0)
    backend: Literal["numpy", "native"] = "numpy"


def main() -> None:
    options = tyro.cli(Options)
    performance = read_midi(Path(__file__).with_name("bwv-578.mid"))
    patches = {
        0: Patch(engine="fm", gain=0.24, pan=-0.2, ratio=1, index=0.65),
        1: Patch(engine="synth", gain=0.20, pan=-0.55),
        2: Patch(engine="sample", gain=0.23, pan=0.45),
        3: Patch(engine="fm", gain=0.20, pan=0.15, ratio=2, index=0.85),
    }
    report = render_midi(
        performance,
        patches,
        options.output,
        options.block_size,
        options.backend,
        progress=True,
    )
    print(
        f"{options.output}: {report.frames / report.sample_rate:.2f}s, "
        f"{report.notes} notes, peak {report.peak:.4f}; "
        f"rendered in {report.elapsed_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
