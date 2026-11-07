"""Prepare individual generator contributions for ordered effect modulation."""

from sys import float_info

from ufor import audio_effects, modulation
from ufor.control import Scope
from ufor.samples.processing import GeneratorBinding

EffectControlRoute = tuple[
    int, int, int, int, list[tuple[float, float]], float, float, str
]
EffectControlPlan = tuple[list[tuple[int, int, float, float]], list[EffectControlRoute]]


def generator_graph(graph: audio_effects.EffectGraph) -> audio_effects.EffectGraph:
    bindings = [b for b in graph.bindings if isinstance(b, GeneratorBinding)]
    names = {b.name for b in bindings}
    routes = [
        r.model_copy(
            update={
                "target": modulation.Target(name=r.name, parameter="value"),
                "unit": modulation.Unit.ratio,
            }
        )
        for r in sorted(graph.modulation.routes, key=lambda r: (r.source, r.name))
        if r.source in names
    ]
    # This is an execution projection, not another portable effect definition.
    # One synthetic target per route preserves contributions until DSP ordering.
    return graph.model_copy(
        update={
            "bindings": bindings,
            "modulation": modulation.Modulation(
                sources=[s for s in graph.modulation.sources if s.name in names],
                routes=routes,
                parameters=[
                    modulation.Parameter(
                        target=r.target,
                        scope=Scope.voice,
                        unit=modulation.Unit.ratio,
                        default=0 if r.operation == modulation.Operation.add else 1,
                        minimum=-float_info.max,
                        maximum=float_info.max,
                    )
                    for r in routes
                ],
            ),
        }
    )
