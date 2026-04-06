import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grader import grade_state
from models import InventoryCurationAction
from server.environment import HyperlocalInventoryCurationEnvironment
from tasks import TASKS



def test_reset_loads_requested_task() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="medium_duplicate_price_fix")
    observation = env.reset()

    assert observation.task_id == "medium_duplicate_price_fix"
    assert observation.done is False
    assert len(observation.records) == len(TASKS["medium_duplicate_price_fix"].records)


def test_reset_clears_state() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="easy_title_unit_cleanup")
    env.reset()

    env.step(
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
        )
    )
    env.step(
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_1",
            field_name="category",
            value="beverages",
        )
    )

    reset_obs = env.reset()

    record = next(r for r in reset_obs.records if r.record_id == "easy_1")
    assert reset_obs.action_history == []
    assert reset_obs.last_action_error is None
    assert record.normalized_title is None
    assert record.category is None


def test_grader_score_is_bounded() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="easy_title_unit_cleanup")
    env.reset()

    breakdown = grade_state(TASKS["easy_title_unit_cleanup"], env.state)

    assert 0.0 <= breakdown.total_score <= 1.0
    assert 0.0 <= breakdown.progress_score <= 1.0


def test_invalid_category_action_gets_penalized() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="medium_duplicate_price_fix")
    env.reset()

    observation = env.step(
        InventoryCurationAction(
            action_type="assign_category",
            record_id="med_3",
            field_name="category",
            value="laundry_detergent",
        )
    )

    assert observation.last_action_error is not None
    assert "invalid category" in observation.last_action_error.lower()
    assert float(observation.reward or 0.0) < 0.0


def test_duplicate_merge_improves_medium_task() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="medium_duplicate_price_fix")
    env.reset()

    before_score = env.state.progress_score

    observation = env.step(
        InventoryCurationAction(
            action_type="merge_duplicate_records",
            record_id="med_1",
            secondary_record_id="med_2",
            reason="Same product in same store with equivalent normalized size.",
        )
    )

    after_score = env.state.progress_score

    assert observation.last_action_error is None
    assert after_score > before_score
    assert ["med_2", "med_1"] in env.state.merged_pairs


def test_flagging_ambiguous_record_improves_hard_task() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="hard_ambiguous_multisource_batch")
    env.reset()

    before_score = env.state.progress_score

    observation = env.step(
        InventoryCurationAction(
            action_type="flag_for_review",
            record_id="hard_4",
            reason="Packaged tomatoes may not be identical to loose tomatoes.",
        )
    )

    after_score = env.state.progress_score

    assert observation.last_action_error is None
    assert after_score > before_score
    assert "hard_4" in env.state.flagged_records


def test_easy_task_reaches_high_score_with_known_good_sequence() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="easy_title_unit_cleanup")
    env.reset()

    good_actions = [
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
        ),
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_2",
            field_name="normalized_title",
            value="Amul Taaza Toned Milk 500 Ml",
        ),
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_3",
            field_name="normalized_title",
            value="Banana 6 Pcs",
        ),
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_4",
            field_name="normalized_title",
            value="Aashirvaad Atta 5 Kg",
        ),
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_5",
            field_name="normalized_title",
            value="Lays Magic Masala 52 G",
        ),
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_1",
            field_name="category",
            value="beverages",
        ),
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_2",
            field_name="category",
            value="dairy",
        ),
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_3",
            field_name="category",
            value="produce",
        ),
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_4",
            field_name="category",
            value="staples",
        ),
        InventoryCurationAction(
            action_type="assign_category",
            record_id="easy_5",
            field_name="category",
            value="snacks",
        ),
    ]

    observation = None
    for action in good_actions:
        observation = env.step(action)

    assert observation is not None
    assert env.state.score >= 0.85


def test_repeated_identical_action_gets_repeat_penalty() -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id="easy_title_unit_cleanup")
    env.reset()

    first = env.step(
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
        )
    )
    second = env.step(
        InventoryCurationAction(
            action_type="normalize_title",
            record_id="easy_1",
            field_name="normalized_title",
            value="Coca Cola 1 L",
        )
    )

    assert float(second.reward or 0.0) <= float(first.reward or 0.0)
