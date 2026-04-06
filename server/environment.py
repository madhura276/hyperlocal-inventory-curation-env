"""
Hyperlocal Inventory Curation Environment.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..grader import build_reward
    from ..models import (
        ActionRecord,
        ActionType,
        Category,
        InventoryCurationAction,
        InventoryCurationObservation,
        InventoryCurationState,
        RecordStatus,
        Unit,
    )
    from ..tasks import DEFAULT_TASK_ID, TASKS
except ImportError:
    from grader import build_reward
    from models import (
        ActionRecord,
        ActionType,
        Category,
        InventoryCurationAction,
        InventoryCurationObservation,
        InventoryCurationState,
        RecordStatus,
        Unit,
    )
    from tasks import DEFAULT_TASK_ID, TASKS


class HyperlocalInventoryCurationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = DEFAULT_TASK_ID):
        self._default_task_id = task_id
        self._task = TASKS[task_id]
        self._state = InventoryCurationState()
        self._seen_action_signatures: set[str] = set()
        self.reset()

    def reset(self) -> InventoryCurationObservation:
        self._task = TASKS[self._default_task_id]
        self._state = InventoryCurationState(
            episode_id=str(uuid4()),
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self._task.max_steps,
            done=False,
            records=deepcopy(self._task.records),
            merged_pairs=[],
            flagged_records=[],
            action_history=[],
            last_action_error=None,
            last_reward=None,
            score=0.0,
            progress_score=0.0,
        )
        self._seen_action_signatures = set()
        return self._build_observation()

    def step(self, action: InventoryCurationAction) -> InventoryCurationObservation:  # type: ignore[override]
        if self._state.done:
            self._state.last_action_error = "Episode already finished. Reset the environment."
            return self._build_observation(reward_override=-0.05)

        previous_state = deepcopy(self._state)
        penalty = 0.0
        error: str | None = None

        try:
            penalty += self._apply_action(action)
        except ValueError as exc:
            error = str(exc)
            penalty += 0.08

        self._state.step_count += 1
        submitted = action.action_type == ActionType.FINALIZE_BATCH

        if submitted or self._state.step_count >= self._state.max_steps:
            self._state.done = True

        reward_model = build_reward(
            task=self._task,
            previous_state=previous_state,
            current_state=self._state,
            penalty=penalty,
            submitted=submitted,
        )
        self._state.last_reward = reward_model
        self._state.last_action_error = error
        self._state.score = reward_model.total_score
        self._state.progress_score = reward_model.progress_score

        self._state.action_history.append(
            ActionRecord(
                step=self._state.step_count,
                action_type=action.action_type,
                record_id=action.record_id,
                secondary_record_id=action.secondary_record_id,
                field_name=action.field_name,
                value=action.value,
                reward=reward_model.delta,
                error=error,
            )
        )

        return self._build_observation()

    @property
    def state(self) -> InventoryCurationState:
        return deepcopy(self._state)

    def _build_observation(self, reward_override: float | None = None) -> InventoryCurationObservation:
        reward_value = reward_override
        if reward_value is None and self._state.last_reward is not None:
            reward_value = self._state.last_reward.delta

        return InventoryCurationObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            records=deepcopy(self._state.records),
            policy_snippets=self._task.policy.policy_snippets,
            action_history=deepcopy(self._state.action_history),
            remaining_steps=max(self._state.max_steps - self._state.step_count, 0),
            last_action_error=self._state.last_action_error,
            reward_details=deepcopy(self._state.last_reward),
            done=self._state.done,
            reward=reward_value if reward_value is not None else 0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "task_title": self._task.title,
                "score": self._state.score,
            },
        )

    def _apply_action(self, action: InventoryCurationAction) -> float:
        penalty = 0.0

        signature = (
            f"{action.action_type.value}|{action.record_id}|{action.secondary_record_id}|"
            f"{action.field_name}|{action.value}|{action.reason}"
        )
        if signature in self._seen_action_signatures and action.action_type != ActionType.FINALIZE_BATCH:
            penalty += 0.03
        self._seen_action_signatures.add(signature)

        if action.action_type == ActionType.NORMALIZE_TITLE:
            record = self._require_record(action.record_id)
            if not isinstance(action.value, str) or len(action.value.strip()) < 3:
                raise ValueError("normalize_title requires a non-empty string value")
            record.normalized_title = action.value.strip()
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif action.action_type == ActionType.NORMALIZE_SIZE:
            record = self._require_record(action.record_id)
            if not action.field_name:
                raise ValueError("normalize_size requires field_name")
            if action.field_name == "quantity_value":
                if not isinstance(action.value, (int, float)) or float(action.value) <= 0:
                    raise ValueError("quantity_value must be > 0")
                record.quantity_value = float(action.value)
            elif action.field_name == "quantity_unit":
                if not isinstance(action.value, str):
                    raise ValueError("quantity_unit must be a string")
                try:
                    record.quantity_unit = Unit(action.value.lower())
                except ValueError as exc:
                    raise ValueError(f"invalid unit: {action.value}") from exc
            elif action.field_name == "pack_count":
                if not isinstance(action.value, (int, float)) or int(action.value) < 1:
                    raise ValueError("pack_count must be >= 1")
                record.pack_count = int(action.value)
            else:
                raise ValueError("normalize_size field_name must be quantity_value, quantity_unit, or pack_count")
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif action.action_type == ActionType.ASSIGN_CATEGORY:
            record = self._require_record(action.record_id)
            if not isinstance(action.value, str):
                raise ValueError("assign_category requires a string value")
            try:
                record.category = Category(action.value.lower())
            except ValueError as exc:
                raise ValueError(f"invalid category: {action.value}") from exc
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif action.action_type == ActionType.MERGE_DUPLICATE_RECORDS:
            primary = self._require_record(action.record_id)
            secondary = self._require_record(action.secondary_record_id)

            if primary.record_id == secondary.record_id:
                raise ValueError("cannot merge a record into itself")
            if primary.store_id != secondary.store_id:
                raise ValueError("cannot merge records from different stores")
            if secondary.status == RecordStatus.MERGED:
                raise ValueError("secondary record is already merged")

            secondary.status = RecordStatus.MERGED
            note = f"Merged into {primary.record_id}"
            if note not in secondary.notes:
                secondary.notes.append(note)

            pair = [secondary.record_id, primary.record_id]
            if pair not in self._state.merged_pairs:
                self._state.merged_pairs.append(pair)

        elif action.action_type == ActionType.CORRECT_PRICE:
            record = self._require_record(action.record_id)
            if not isinstance(action.value, (int, float)) or float(action.value) < 0:
                raise ValueError("correct_price requires a numeric value >= 0")
            record.price = float(action.value)
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif action.action_type == ActionType.FILL_MISSING_ATTRIBUTE:
            record = self._require_record(action.record_id)
            if not action.field_name:
                raise ValueError("fill_missing_attribute requires field_name")
            allowed_fields = {"brand", "subcategory", "barcode"}
            if action.field_name not in allowed_fields:
                raise ValueError(f"fill_missing_attribute only supports: {', '.join(sorted(allowed_fields))}")
            if not isinstance(action.value, str) or not action.value.strip():
                raise ValueError("fill_missing_attribute requires a non-empty string value")
            setattr(record, action.field_name, action.value.strip())
            if record.status == RecordStatus.RAW:
                record.status = RecordStatus.CLEANED

        elif action.action_type == ActionType.FLAG_FOR_REVIEW:
            record = self._require_record(action.record_id)
            record.status = RecordStatus.FLAGGED
            if action.reason and action.reason.strip():
                record.notes.append(f"Flagged: {action.reason.strip()}")
            if record.record_id not in self._state.flagged_records:
                self._state.flagged_records.append(record.record_id)

        elif action.action_type == ActionType.FINALIZE_BATCH:
            unresolved = [
                record.record_id
                for record in self._state.records
                if record.status == RecordStatus.RAW
            ]
            if unresolved:
                penalty += min(0.12, 0.02 * len(unresolved))

        else:
            raise ValueError(f"unsupported action_type: {action.action_type}")

        return penalty

    def _require_record(self, record_id: str | None):
        if not record_id:
            raise ValueError("record_id is required for this action")
        for record in self._state.records:
            if record.record_id == record_id:
                return record
        raise ValueError(f"record not found: {record_id}")
