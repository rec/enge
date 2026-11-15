"""Summarize native callback heap counters recorded in pytest JUnit reports."""

from ast import literal_eval
from pathlib import Path
from typing import Annotated
from xml.etree import ElementTree

import tyro
from pydantic import BaseModel, ConfigDict, Field


class CallbackHeap(BaseModel, frozen=True):
    successful: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        min_length=4, max_length=4
    )
    failed: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        min_length=4, max_length=4
    )

    model_config = ConfigDict(extra="forbid")


class AuditSummary(CallbackHeap, frozen=True):
    tests: int
    native_call: CallbackHeap


class Options(BaseModel, frozen=True):
    reports: list[Path] = Field(min_length=1)


def summarize(reports: list[Path]) -> AuditSummary:
    tests = 0
    totals = {
        p: {n: [0, 0, 0, 0] for n in ("successful", "failed")}
        for p in ("native_callback_heap", "native_call_heap")
    }
    for report in reports:
        root = ElementTree.parse(report)
        tests += len(root.findall(".//testcase"))
        for property, boundary in totals.items():
            properties = root.findall(f".//property[@name='{property}']")
            if not properties:
                raise ValueError(f"No {property} properties in {report}")
            for item in properties:
                heap = CallbackHeap.model_validate(literal_eval(item.attrib["value"]))
                for name, counts in (
                    ("successful", heap.successful),
                    ("failed", heap.failed),
                ):
                    boundary[name] = [
                        a + b for a, b in zip(boundary[name], counts, strict=True)
                    ]
    return AuditSummary(
        tests=tests,
        successful=totals["native_callback_heap"]["successful"],
        failed=totals["native_callback_heap"]["failed"],
        native_call=CallbackHeap.model_validate(totals["native_call_heap"]),
    )


if __name__ == "__main__":
    print(summarize(tyro.cli(Options).reports).model_dump_json(indent=2))
