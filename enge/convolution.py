"""Prepared matrix impulses and an independent NumPy partitioned FIR reference."""

from hashlib import sha256
from math import isfinite

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects

from .sampler import PreparedSample
from .synth import EngineError


class PreparedKernel(BaseModel, frozen=True):
    input: int
    output: int
    spectra: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PreparedConvolution(BaseModel, frozen=True):
    partition: int
    inputs: int
    outputs: int
    parts: int
    tail_frames: int
    kernels: list[PreparedKernel]


class ConvolutionState(BaseModel):
    input_block: list[list[float]]
    output_block: list[list[float]]
    overlap: list[list[float]]
    spectra_real: list[list[list[float]]]
    spectra_imag: list[list[list[float]]]
    fill: int = Field(default=0, ge=0)
    slot: int = Field(default=0, ge=0)
    input_ended: bool = False
    tail_remaining: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def prepare_audio(
    graph: audio_effects.EffectGraph,
    sample_rate: int,
    audio: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    required = {
        k.asset
        for p in graph.processors
        if isinstance(p, audio_effects.Convolution)
        for k in p.kernels
    }
    supplied = {} if audio is None else audio
    if supplied.keys() != required:
        raise EngineError("Decoded impulses must match referenced asset IDs")
    assets = {a.name: a for a in graph.assets}
    rates = {t.name: t.rate for t in graph.timebases}
    buffers: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in sorted(required):
        metadata = assets[name].audio
        rate = rates[metadata.timebase]
        if rate.numerator != sample_rate * rate.denominator:
            raise EngineError("Impulse sample rate must match the effect render rate")
        if supplied[name].shape != (metadata.frames, len(metadata.channels)):
            raise EngineError(
                f"Decoded impulse {name} differs from its declared extent/layout"
            )
        buffers[name] = PreparedSample.immutable_audio(supplied[name])
        digests[name] = sha256(
            buffers[name].astype("<f8", copy=False).tobytes()
        ).hexdigest()
    return buffers, digests


def prepare(
    processor: audio_effects.Convolution,
    graph: audio_effects.EffectGraph,
    audio: dict[str, np.ndarray],
) -> PreparedConvolution:
    assets = {a.name: a for a in graph.assets}
    block = processor.partition_frames
    if block > np.iinfo(np.intp).max // 2 or int(float(block)) != block:
        raise EngineError("Convolution partition cannot use the native frame format")
    maximum = max(len(audio[k.asset]) for k in processor.kernels)
    cache: dict[tuple[str, str], np.ndarray] = {}
    kernels: list[PreparedKernel] = []
    for kernel in processor.kernels:
        key = kernel.asset, kernel.channel
        if key not in cache:
            values = audio[kernel.asset][
                :, assets[kernel.asset].audio.channels.index(kernel.channel)
            ]
            padded = np.zeros(((len(values) + block - 1) // block) * block)
            padded[: len(values)] = values
            spectra = np.fft.rfft(padded.reshape(-1, block), n=2 * block, axis=1)
            if not np.all(np.isfinite(spectra)):
                raise EngineError("Non-finite impulse spectrum")
            cache[key] = np.frombuffer(spectra.tobytes(), dtype=np.complex128).reshape(
                spectra.shape
            )
        kernels.append(
            PreparedKernel(
                input=processor.input_channels.index(kernel.input_channel),
                output=processor.output_channels.index(kernel.output_channel),
                spectra=cache[key],
            )
        )
    return PreparedConvolution(
        partition=block,
        inputs=len(processor.input_channels),
        outputs=len(processor.output_channels),
        parts=(maximum + block - 1) // block,
        tail_frames=maximum + block - 1,
        kernels=kernels,
    )


def initial_state(parameters: PreparedConvolution) -> ConvolutionState:
    block = parameters.partition
    return ConvolutionState(
        input_block=[[0.0] * parameters.inputs for _ in range(block)],
        output_block=[[0.0] * parameters.outputs for _ in range(block)],
        overlap=[[0.0] * parameters.outputs for _ in range(block)],
        spectra_real=[
            [[0.0] * (block + 1) for _ in range(parameters.inputs)]
            for _ in range(parameters.parts)
        ],
        spectra_imag=[
            [[0.0] * (block + 1) for _ in range(parameters.inputs)]
            for _ in range(parameters.parts)
        ],
    )


def validate_state(state: ConvolutionState, parameters: PreparedConvolution) -> None:
    block = parameters.partition
    matrices = [
        (state.input_block, parameters.inputs),
        (state.output_block, parameters.outputs),
        (state.overlap, parameters.outputs),
    ]
    if any(
        len(m) != block
        or any(len(r) != n or not all(isfinite(v) for v in r) for r in m)
        for m, n in matrices
    ):
        raise EngineError("Invalid convolution block state")
    for cube in (state.spectra_real, state.spectra_imag):
        if len(cube) != parameters.parts or any(
            len(m) != parameters.inputs
            or any(len(r) != block + 1 or not all(isfinite(v) for v in r) for r in m)
            for m in cube
        ):
            raise EngineError("Invalid convolution spectrum history")
    if any(r[0] != 0 or r[-1] != 0 for m in state.spectra_imag for r in m):
        raise EngineError("Convolution DC/Nyquist bins must be real")
    if (
        not 0 <= state.fill < block
        or not 0 <= state.slot < parameters.parts
        or not 0 <= state.tail_remaining <= parameters.tail_frames
        or (not state.input_ended and state.tail_remaining != 0)
    ):
        raise EngineError("Invalid convolution partition clock")


def process(
    state: ConvolutionState,
    parameters: PreparedConvolution,
    source: np.ndarray,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    if state.input_ended and state.tail_remaining == 0:
        return np.zeros(parameters.outputs), True
    if ended and not state.input_ended:
        state.input_ended = True
        state.tail_remaining = parameters.tail_frames
    wet = np.asarray(state.output_block[state.fill])
    state.input_block[state.fill] = (
        np.zeros(parameters.inputs) if state.input_ended else source
    ).tolist()
    state.fill += 1
    if state.fill == parameters.partition:
        state.fill = 0
        _finish_block(state, parameters)
    if state.input_ended:
        state.tail_remaining -= 1
    return wet, False


def _finish_block(state: ConvolutionState, parameters: PreparedConvolution) -> None:
    block = parameters.partition
    spectrum = np.fft.rfft(np.asarray(state.input_block), n=2 * block, axis=0).T
    if not np.all(np.isfinite(spectrum)):
        raise EngineError("Non-finite convolution input spectrum")
    state.spectra_real[state.slot] = spectrum.real.tolist()
    state.spectra_imag[state.slot] = spectrum.imag.tolist()
    history = np.asarray(state.spectra_real) + 1j * np.asarray(state.spectra_imag)
    total = np.zeros((parameters.outputs, block + 1), dtype=np.complex128)
    for kernel in parameters.kernels:
        for part in range(len(kernel.spectra)):
            total[kernel.output] += (
                history[(state.slot - part) % parameters.parts, kernel.input]
                * kernel.spectra[part]
            )
    result = np.fft.irfft(total, n=2 * block, axis=1).T
    output = result[:block] + np.asarray(state.overlap)
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(result)):
        raise EngineError("Non-finite convolution output")
    state.output_block = output.tolist()
    state.overlap = result[block:].tolist()
    state.slot = (state.slot + 1) % parameters.parts


type NativeConvolutionPlan = list[tuple[int, list[tuple[int, int, np.ndarray]]]]
