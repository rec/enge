"""Materialize audio-host bindings into one standalone scheduling graph."""

from envelfo import patch
from ufor import envelfo, envelfo_random
from ufor.samples import processing


def prepare_graph(settings: processing.SoundSettings) -> patch.PreparedPatch:
    children: dict[str, object] = {}
    outputs: dict[str, str] = {}
    inputs: list[str] = []
    sample_keys: dict[str, int] = {}
    bindings = {
        b.name: b
        for b in settings.bindings
        if isinstance(b, processing.GeneratorBinding)
    }
    patch_children = {
        f"patch-{n}-{c}"
        for n, p in settings.envelfos.items()
        if isinstance(p, envelfo.Patch)
        for c, b in p.envelfos.items()
        if isinstance(b, envelfo.EnveLFO)
    }
    for binding in bindings.values():
        body = settings.envelfos[binding.reference]
        if isinstance(body, envelfo.Patch):
            continue
        node = (
            binding.reference
            if binding.reference in patch_children
            else f"lfo-{binding.reference}"
            if isinstance(body, envelfo.LFO)
            else f"source-{binding.name}"
        )
        children[node] = body
        outputs[binding.name] = node
        if isinstance(body, envelfo.LFO) and body.scope != "voice":
            if node not in inputs:
                inputs.append(node)
        if isinstance(body, envelfo.SampleHold):
            sample_keys[node] = envelfo_random.stream_key(
                0, binding.reference + ":sample-hold"
            )
    for parent_name, parent in settings.envelfos.items():
        if not isinstance(parent, envelfo.Patch):
            continue
        selected = [b for b in bindings.values() if b.reference == parent_name]
        if not selected:
            continue
        for name in parent.signal_order:
            body = parent.envelfos[name]
            node = f"patch-{parent_name}-{name}"
            if isinstance(body, (envelfo.Sum, envelfo.Product)):
                if any(f"patch-{parent_name}-{n}" not in children for n in body.inputs):
                    continue
                children[node] = body.model_copy(
                    update={"inputs": [f"patch-{parent_name}-{n}" for n in body.inputs]}
                )
            elif isinstance(
                body,
                (
                    envelfo.Affine,
                    envelfo.Quantize,
                    envelfo.Latch,
                    envelfo.Threshold,
                    envelfo.Slew,
                ),
            ):
                if f"patch-{parent_name}-{body.input}" not in children:
                    continue
                children[node] = body.model_copy(
                    update={"input": f"patch-{parent_name}-{body.input}"}
                )
        for binding in selected:
            assert binding.output is not None
            outputs[binding.name] = (
                f"patch-{parent_name}-{parent.outputs[binding.output]}"
            )
    events: list[envelfo.PatchEventConnection] = []
    identities: list[str] = []
    for parent_name, parent in settings.envelfos.items():
        if not isinstance(parent, envelfo.Patch):
            continue
        for order, connection in enumerate(parent.events):
            child_name, _, port = connection.source.partition(".")
            threshold = isinstance(parent.envelfos[child_name], envelfo.Threshold)
            if connection.action == "cue" and not threshold:
                # Preparation already materialized these as voice connections.
                continue
            source = f"patch-{parent_name}-{child_name}"
            target = f"patch-{parent_name}-{connection.target}"
            if source not in children or target not in children:
                continue
            events.append(
                connection.model_copy(
                    update={"source": f"{source}.{port}", "target": target}
                )
            )
            identities.append(
                f"{parent_name}:capture-{connection.target}-{order}"
                if connection.action == "capture"
                else f"{target}:cue-{order}"
                if connection.action == "cue"
                else f"{target}:start-{order}"
            )
    for order, connection in enumerate(settings.event_connections):
        source_binding = bindings[connection.source]
        source_body = settings.envelfos[source_binding.reference]
        if isinstance(source_body, envelfo.Patch):
            child, _, port = source_body.event_outputs[connection.port].partition(".")
            source = f"patch-{source_binding.reference}-{child}"
        else:
            source, port = outputs[connection.source], connection.port
        target = outputs[connection.destination]
        events.append(
            envelfo.PatchEventConnection(
                source=f"{source}.{port}",
                target=target,
                action="cue",
                cue=connection.cue,
                every=connection.every,
                offset=connection.offset,
                probability=connection.probability,
                delay=connection.delay,
            )
        )
        identities.append(f"{bindings[connection.destination].reference}:cue-{order}")
    return patch.PreparedPatch.model_validate(
        {
            "envelfos": children,
            "outputs": outputs,
            "events": events,
            "inputs": inputs,
            "sample_keys": sample_keys,
            "connection_identities": identities,
        }
    )
