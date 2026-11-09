from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts.check_wt11_reference import check, compare_rates, rate_table
from scripts.summarize_callback_audit import summarize


def test_callback_summary_combines_reports_and_retains_nonzero_heap_counts(
    tmp_path: Path,
) -> None:
    reports = []
    for i in range(2):
        root = ElementTree.Element("testsuites")
        suite = ElementTree.SubElement(root, "testsuite")
        for _ in range(2):
            case = ElementTree.SubElement(suite, "testcase")
            properties = ElementTree.SubElement(case, "properties")
            ElementTree.SubElement(
                properties,
                "property",
                name="native_callback_heap",
                value=str({"successful": (3, 0, 0, 0), "failed": (1, 2, 3, 4)}),
            )
        report = tmp_path / f"audit-{i}.xml"
        ElementTree.ElementTree(root).write(report)
        reports.append(report)
    summary = summarize(reports)
    assert summary.tests == 4
    assert summary.successful == [12, 0, 0, 0]
    assert summary.failed == [4, 8, 12, 16]


def test_callback_summary_rejects_reports_without_heap_measurements(
    tmp_path: Path,
) -> None:
    report = tmp_path / "ordinary.xml"
    report.write_text("<testsuites><testsuite><testcase /></testsuite></testsuites>")
    with pytest.raises(ValueError, match="No native_callback_heap"):
        summarize([report])


@pytest.mark.parametrize("rate", [0, 50, 99])
def test_wt11_reference_check_detects_mismatches_throughout_legal_range(
    rate: int,
) -> None:
    reference = list(range(101))
    compare_rates(reference, reference[:100])
    actual = reference[:100]
    actual[rate] += 1
    with pytest.raises(ValueError, match=f"rates: {rate}$"):
        compare_rates(reference, actual)
    with pytest.raises(ValueError, match="exactly 100"):
        compare_rates(reference, actual[:-1])


def test_reference_table_is_read_as_data_without_executing_source(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text('raise RuntimeError("must not run")\npr = (1, 2, 3)\n')
    assert rate_table(reference, "pr") == [1, 2, 3]
    with pytest.raises(ValueError, match="Missing table"):
        rate_table(reference, "missing")
    with pytest.raises(ValueError, match="DXconvert revision"):
        check(reference)
