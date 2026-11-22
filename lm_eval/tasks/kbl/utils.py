"""Shared preprocessing for KBL tasks."""

from __future__ import annotations


BAR_EXAM_ANSWERS = frozenset({"A", "B", "C", "D", "E"})


def process_bar_exam_docs(dataset):
    """Drop bar-exam rows whose gold answer is not an available A-E choice.

    The KBL dataset contains at least one ``gt='정답없음'`` row while the
    benchmark protocol exposes only five choices. Such rows cannot form a
    valid target and must not abort the whole benchmark.
    """
    return dataset.filter(
        lambda doc: doc.get("gt") in BAR_EXAM_ANSWERS,
        desc="Filtering KBL bar-exam rows with no A-E gold answer",
    )
