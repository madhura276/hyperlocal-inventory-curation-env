from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from models import (
    Category,
    ExpectedRecordOutcome,
    InventoryCurationReward,
    InventoryCurationState,
    InventoryRecord,
    RecordStatus,
    RewardComponent,
    TaskDefinition,
    Unit,
)


TITLE_WEIGHT = 0.20
SIZE_WEIGHT = 0.20
CATEGORY_WEIGHT = 0.20
DUPLICATE_WEIGHT = 0.15
PRICE_WEIGHT = 0.15
ESCALATION_WEIGHT = 0.10


@dataclass(frozen=True)
class GradeBreakdown:
    total_score: float
    progress_score: float
    components: List[RewardComponent]


SCORE_EPS = 1e-3

def _clip_score(value: float) -> float:
    return max(SCORE_EPS, min(1.0 - SCORE_EPS, value))


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _float_equal(a: Optional[float], b: Optional[float], tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def _build_record_map(records: Iterable[InventoryRecord]) -> Dict[str, InventoryRecord]:
    return {record.record_id: record for record in records}


def _build_expected_map(task: TaskDefinition) -> Dict[str, ExpectedRecordOutcome]:
    return {outcome.record_id: outcome for outcome in task.expected_outcomes}


def score_title_normalization(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.normalized_title is not None
    ]
    if not expected_ids:
        return 1.0

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and _normalize_text(actual.normalized_title) == _normalize_text(expected.normalized_title):
            hits += 1

    return hits / len(expected_ids)


def score_unit_normalization(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.quantity_value is not None or outcome.quantity_unit is not None or outcome.pack_count is not None
    ]
    if not expected_ids:
        return 1.0

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if not actual:
            continue

        quantity_ok = True
        unit_ok = True
        pack_ok = True

        if expected.quantity_value is not None:
            quantity_ok = _float_equal(actual.quantity_value, expected.quantity_value)

        if expected.quantity_unit is not None:
            unit_ok = actual.quantity_unit == expected.quantity_unit

        if expected.pack_count is not None:
            pack_ok = actual.pack_count == expected.pack_count

        if quantity_ok and unit_ok and pack_ok:
            hits += 1

    return hits / len(expected_ids)


def score_category_assignment(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.category is not None
    ]
    if not expected_ids:
        return 1.0

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and actual.category == expected.category:
            hits += 1

    return hits / len(expected_ids)


def score_duplicate_resolution(task: TaskDefinition, state: InventoryCurationState) -> float:
    expected_pairs = {
        (outcome.record_id, outcome.merged_into)
        for outcome in task.expected_outcomes
        if outcome.merged_into is not None
    }
    actual_pairs = {
        (pair[0], pair[1])
        for pair in state.merged_pairs
        if len(pair) == 2
    }

    if not expected_pairs:
        return 1.0 if not actual_pairs else 0.0

    correct_pairs = expected_pairs & actual_pairs
    extra_pairs = actual_pairs - expected_pairs

    base = len(correct_pairs) / len(expected_pairs)
    penalty = len(extra_pairs) / max(len(expected_pairs), 1)
    return _clip_score(base - 0.5 * penalty)


def score_price_handling(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.price is not None
    ]
    if not expected_ids:
        return 1.0

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and _float_equal(actual.price, expected.price):
            hits += 1

    return hits / len(expected_ids)


def score_escalation_quality(task: TaskDefinition, state: InventoryCurationState) -> float:
    expected_flagged = {
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.should_flag
    }
    actual_flagged = set(state.flagged_records)

    record_map = _build_record_map(state.records)
    for record_id, record in record_map.items():
        if record.status == RecordStatus.FLAGGED:
            actual_flagged.add(record_id)

    if not expected_flagged:
        return 1.0 if not actual_flagged else 0.0

    correct_flags = expected_flagged & actual_flagged
    extra_flags = actual_flagged - expected_flagged

    base = len(correct_flags) / len(expected_flagged)
    penalty = len(extra_flags) / max(len(expected_flagged), 1)
    return _clip_score(base - 0.5 * penalty)


def score_status_alignment(task: TaskDefinition, records: List[InventoryRecord]) -> float:
    expected_map = _build_expected_map(task)
    record_map = _build_record_map(records)

    expected_ids = [
        outcome.record_id
        for outcome in task.expected_outcomes
        if outcome.status is not None
    ]
    if not expected_ids:
        return 1.0

    hits = 0
    for record_id in expected_ids:
        expected = expected_map[record_id]
        actual = record_map.get(record_id)
        if actual and actual.status == expected.status:
            hits += 1

    return hits / len(expected_ids)


def grade_state(task: TaskDefinition, state: InventoryCurationState) -> GradeBreakdown:
    title_score = score_title_normalization(task, state.records)
    size_score = score_unit_normalization(task, state.records)
    category_score = score_category_assignment(task, state.records)
    duplicate_score = score_duplicate_resolution(task, state)
    price_score = score_price_handling(task, state.records)
    escalation_score = score_escalation_quality(task, state)
    status_score = score_status_alignment(task, state.records)

    components = [
        RewardComponent(
            name="title_normalization",
            score=title_score,
            weight=TITLE_WEIGHT,
            contribution=TITLE_WEIGHT * title_score,
        ),
        RewardComponent(
            name="size_normalization",
            score=size_score,
            weight=SIZE_WEIGHT,
            contribution=SIZE_WEIGHT * size_score,
        ),
        RewardComponent(
            name="category_assignment",
            score=category_score,
            weight=CATEGORY_WEIGHT,
            contribution=CATEGORY_WEIGHT * category_score,
        ),
        RewardComponent(
            name="duplicate_resolution",
            score=duplicate_score,
            weight=DUPLICATE_WEIGHT,
            contribution=DUPLICATE_WEIGHT * duplicate_score,
        ),
        RewardComponent(
            name="price_handling",
            score=price_score,
            weight=PRICE_WEIGHT,
            contribution=PRICE_WEIGHT * price_score,
        ),
        RewardComponent(
            name="escalation_quality",
            score=escalation_score,
            weight=ESCALATION_WEIGHT,
            contribution=ESCALATION_WEIGHT * escalation_score,
        ),
    ]

    progress_score = _clip_score(sum(component.contribution for component in components))
    total_score = _clip_score((0.85 * progress_score) + (0.15 * status_score))

    return GradeBreakdown(
        total_score=total_score,
        progress_score=progress_score,
        components=components,
    )


def build_reward(
    task: TaskDefinition,
    previous_state: InventoryCurationState,
    current_state: InventoryCurationState,
    penalty: float = 0.0,
    submitted: bool = False,
) -> InventoryCurationReward:
    previous_grade = grade_state(task, previous_state)
    current_grade = grade_state(task, current_state)

    delta = current_grade.progress_score - previous_grade.progress_score - penalty
    if submitted and current_grade.total_score >= 0.85:
        delta += 0.05

    explanation_parts = [
        f"progress {previous_grade.progress_score:.2f}->{current_grade.progress_score:.2f}",
        f"total {current_grade.total_score:.2f}",
    ]
    if penalty > 0:
        explanation_parts.append(f"penalty {penalty:.2f}")
    if submitted:
        explanation_parts.append("batch finalized")

    return InventoryCurationReward(
        delta=round(delta, 4),
        total_score=round(current_grade.total_score, 4),
        progress_score=round(current_grade.progress_score, 4),
        penalty=round(penalty, 4),
        components=current_grade.components,
        explanation="; ".join(explanation_parts),
    )
