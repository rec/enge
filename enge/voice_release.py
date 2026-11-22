"""Use Rust uFor to plan key releases and sustain changes."""

from ufor.voice_release import ReleasePlan, ReleaseRequest

from . import _native


def native_release_plan(request: ReleaseRequest) -> ReleasePlan:
    document = request.model_dump_json(
        include={
            "event": True,
            "triggers": {
                "__all__": {
                    "part",
                    "trigger_id",
                    "physically_released",
                    "logical_released",
                }
            },
        }
    )
    return ReleasePlan.model_validate_json(_native.plan_voice_release(document))
