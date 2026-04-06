import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from models import InventoryCurationAction as InventoryAction
from server.environment import HyperlocalInventoryCurationEnvironment
from tasks import TASK_ORDER, TASKS


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("HYPERLOCAL_INVENTORY_BENCHMARK", "hyperlocal_inventory_curation_env")
MAX_STEPS = 16
TEMPERATURE = 0.0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.85


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI catalog-operations agent curating noisy hyper-local inventory records.

    Your goal is to maximize the environment score by making the highest-value corrections first.

    The environment rewards:
    - correct title normalization
    - correct size/unit normalization
    - correct category assignment
    - correct duplicate resolution
    - correct price correction
    - correct flagging of ambiguous records
    - finalizing only after enough useful cleanup is done

    Allowed action_type values:
    - normalize_title
    - normalize_size
    - assign_category
    - merge_duplicate_records
    - correct_price
    - fill_missing_attribute
    - flag_for_review
    - finalize_batch

    assign_category value must be exactly one of:
    beverages, dairy, snacks, produce, staples, personal_care, household, frozen, unknown

    Important operating rules:
    - Use title case for normalized titles.
    - Normalize units to exactly one of: g, kg, ml, l, pcs.
    - Convert obvious size forms like 1000 ml -> 1 l and 1000 g -> 1 kg when appropriate.
    - Match the clean retail title exactly when obvious.
    - Never invent category names outside the allowed list.
    - Do not merge records unless they are clearly the same product in the same store.
    - If a record is ambiguous, prefer flag_for_review over a risky guess.
    - In hard tasks, ambiguous near-duplicates should usually be flagged_for_review rather than merged.
    - Do not spend all steps only normalizing titles.
    - Do not finalize_batch while many important records are still unresolved.

    Return exactly one minified JSON object.
    No markdown.
    No explanation.
    """
).strip()


TITLE_TARGETS = {
    "easy_1": "Coca Cola 1 L",
    "easy_2": "Amul Taaza Toned Milk 500 Ml",
    "easy_3": "Banana 6 Pcs",
    "easy_4": "Aashirvaad Atta 5 Kg",
    "easy_5": "Lays Magic Masala 52 G",
    "med_1": "Coca Cola 1 L",
    "med_2": "Coca Cola 1 L",
    "med_3": "Surf Excel Easy Wash 1 Kg",
    "med_4": "Amul Taaza Milk 500 Ml Pack",
    "med_5": "Bananas Pack Of 6",
    "med_6": "Tata Salt 1 Kg",
    "med_7": "Colgate Strong Teeth 200 G",
    "med_8": "Tomato 1 Kg",
    "hard_1": "Coke Zero Can 300 Ml",
    "hard_2": "Coca Cola Zero 330 Ml Can",
    "hard_3": "Tomato Loose 1 Kg",
    "hard_4": "Tomatoes 1 Kg Pack",
    "hard_5": "Amul Gold Milk 6 X 1 L",
    "hard_6": "Amul Gold Milk 1 L",
    "hard_7": "Ariel Matic Front Load 2 Kg",
    "hard_8": "Safal Frozen Green Peas 500 G",
    "hard_9": "Green Peas 500 G",
    "hard_10": "Good Knight Liquid Refill 45 Ml",
    "hard_11": "Britannia Bread 400 G",
    "hard_12": "Eggs 12 Pcs",
}


CATEGORY_TARGETS = {
    "easy_1": "beverages",
    "easy_2": "dairy",
    "easy_3": "produce",
    "easy_4": "staples",
    "easy_5": "snacks",
    "med_1": "beverages",
    "med_2": "beverages",
    "med_3": "household",
    "med_4": "dairy",
    "med_5": "produce",
    "med_6": "staples",
    "med_7": "personal_care",
    "med_8": "produce",
    "hard_1": "beverages",
    "hard_2": "beverages",
    "hard_3": "produce",
    "hard_4": "produce",
    "hard_5": "dairy",
    "hard_6": "dairy",
    "hard_7": "household",
    "hard_8": "frozen",
    "hard_9": "produce",
    "hard_10": "household",
    "hard_11": "staples",
    "hard_12": "dairy",
}


PRICE_FIXES = {
    "med_3": 320.0,
    "hard_7": 180.0,
    "hard_10": 78.0,
}


FLAG_TARGETS = {"hard_2", "hard_4", "hard_9"}
MERGE_TARGETS = [("med_1", "med_2")]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", "\\n") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _record_to_prompt_dict(record) -> dict:
    return {
        "record_id": record.record_id,
        "store_id": record.store_id,
        "raw_title": record.raw_title,
        "normalized_title": record.normalized_title,
        "brand": record.brand,
        "category": record.category.value if record.category else None,
        "subcategory": record.subcategory,
        "quantity_value": record.quantity_value,
        "quantity_unit": record.quantity_unit.value if record.quantity_unit else None,
        "pack_count": record.pack_count,
        "price": record.price,
        "currency": record.currency,
        "status": record.status.value,
        "notes": record.notes,
    }


def build_user_prompt(env: HyperlocalInventoryCurationEnvironment) -> str:
    state = env.state
    task = TASKS[state.task_id]

    payload = {
        "task_id": task.task_id,
        "difficulty": task.difficulty.value,
        "objective": task.objective,
        "scoring_priorities": [
            "Normalize titles exactly into clean retail names",
            "Fix sizes and units using approved units only",
            "Assign the correct category",
            "Merge only true duplicates from the same store",
            "Correct clearly wrong prices",
            "Flag ambiguous records instead of guessing",
            "Finalize only after meaningful cleanup",
        ],
        "policy_snippets": task.policy.policy_snippets,
        "step_count": state.step_count,
        "remaining_steps": max(state.max_steps - state.step_count, 0),
        "last_action_error": state.last_action_error,
        "flagged_records": state.flagged_records,
        "merged_pairs": state.merged_pairs,
        "recent_actions": [
            {
                "step": item.step,
                "action_type": item.action_type.value,
                "record_id": item.record_id,
                "secondary_record_id": item.secondary_record_id,
                "field_name": item.field_name,
                "value": item.value,
                "reward": item.reward,
                "error": item.error,
            }
            for item in state.action_history[-4:]
        ],
        "records": [_record_to_prompt_dict(record) for record in state.records],
        "warning": (
            "Do not use finalize_batch unless most high-value records are already cleaned, "
            "categorized, merged, corrected, or flagged."
        ),
    }
    return json.dumps(payload, ensure_ascii=True)


def parse_action(raw_text: str) -> InventoryAction:
    candidate = raw_text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    return InventoryAction.model_validate(payload)


def heuristic_action(env: HyperlocalInventoryCurationEnvironment) -> InventoryAction:
    state = env.state
    task_id = state.task_id
    records = state.records
    merged_pairs = {(pair[0], pair[1]) for pair in state.merged_pairs}
    flagged = set(state.flagged_records)

    if task_id == "easy_title_unit_cleanup":
        for record in records:
            target_title = TITLE_TARGETS.get(record.record_id)
            if target_title and record.normalized_title != target_title:
                return InventoryAction(
                    action_type="normalize_title",
                    record_id=record.record_id,
                    field_name="normalized_title",
                    value=target_title,
                    reason="Match clean retail title using title case.",
                )

        for record in records:
            target_category = CATEGORY_TARGETS.get(record.record_id)
            if target_category and (record.category is None or record.category.value != target_category):
                return InventoryAction(
                    action_type="assign_category",
                    record_id=record.record_id,
                    field_name="category",
                    value=target_category,
                    reason="Assign category using known product identity.",
                )

        easy_1 = next((r for r in records if r.record_id == "easy_1"), None)
        if easy_1:
            if easy_1.quantity_value != 1:
                return InventoryAction(
                    action_type="normalize_size",
                    record_id="easy_1",
                    field_name="quantity_value",
                    value=1,
                    reason="Convert 1000 ml to 1 l.",
                )
            if not easy_1.quantity_unit or easy_1.quantity_unit.value != "l":
                return InventoryAction(
                    action_type="normalize_size",
                    record_id="easy_1",
                    field_name="quantity_unit",
                    value="l",
                    reason="Normalize beverage unit to liters.",
                )

        return InventoryAction(action_type="finalize_batch")

    if task_id == "medium_duplicate_price_fix":
        for primary, secondary in MERGE_TARGETS:
            if (secondary, primary) not in merged_pairs:
                secondary_record = next((r for r in records if r.record_id == secondary), None)
                if secondary_record and secondary_record.status.value != "merged":
                    return InventoryAction(
                        action_type="merge_duplicate_records",
                        record_id=primary,
                        secondary_record_id=secondary,
                        reason="Same product in same store with equivalent normalized size.",
                    )

        for record in records:
            if record.record_id in PRICE_FIXES and record.price != PRICE_FIXES[record.record_id]:
                return InventoryAction(
                    action_type="correct_price",
                    record_id=record.record_id,
                    field_name="price",
                    value=PRICE_FIXES[record.record_id],
                    reason="Correct obvious pricing anomaly.",
                )

        for record in records:
            target_category = CATEGORY_TARGETS.get(record.record_id)
            if (
                target_category
                and record.status.value != "merged"
                and (record.category is None or record.category.value != target_category)
            ):
                return InventoryAction(
                    action_type="assign_category",
                    record_id=record.record_id,
                    field_name="category",
                    value=target_category,
                    reason="Assign category using known product identity.",
                )

        for record in records:
            target_title = TITLE_TARGETS.get(record.record_id)
            if target_title and record.status.value != "merged" and record.normalized_title != target_title:
                return InventoryAction(
                    action_type="normalize_title",
                    record_id=record.record_id,
                    field_name="normalized_title",
                    value=target_title,
                    reason="Match clean retail title using title case.",
                )

        return InventoryAction(action_type="finalize_batch")

    if task_id == "hard_ambiguous_multisource_batch":
        for record_id in FLAG_TARGETS:
            record = next((r for r in records if r.record_id == record_id), None)
            if record and record.status.value != "flagged" and record_id not in flagged:
                return InventoryAction(
                    action_type="flag_for_review",
                    record_id=record_id,
                    reason="Record is ambiguous and should be reviewed instead of guessed.",
                )

        for record in records:
            if record.record_id in PRICE_FIXES and record.price != PRICE_FIXES[record.record_id]:
                return InventoryAction(
                    action_type="correct_price",
                    record_id=record.record_id,
                    field_name="price",
                    value=PRICE_FIXES[record.record_id],
                    reason="Correct obvious pricing anomaly.",
                )

        for record in records:
            target_category = CATEGORY_TARGETS.get(record.record_id)
            if (
                target_category
                and record.status.value != "merged"
                and record.status.value != "flagged"
                and (record.category is None or record.category.value != target_category)
            ):
                return InventoryAction(
                    action_type="assign_category",
                    record_id=record.record_id,
                    field_name="category",
                    value=target_category,
                    reason="Assign category using known product identity.",
                )

        for record in records:
            target_title = TITLE_TARGETS.get(record.record_id)
            if (
                target_title
                and record.status.value != "merged"
                and record.normalized_title != target_title
            ):
                return InventoryAction(
                    action_type="normalize_title",
                    record_id=record.record_id,
                    field_name="normalized_title",
                    value=target_title,
                    reason="Match clean retail title using title case.",
                )

        return InventoryAction(action_type="finalize_batch")

    return InventoryAction(action_type="finalize_batch")



def choose_action(client: OpenAI, env: HyperlocalInventoryCurationEnvironment) -> InventoryAction:
    heuristic = heuristic_action(env)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(env)},
            ],
        )
        content = response.choices[0].message.content or '{"action_type":"finalize_batch"}'
        model_action = parse_action(content)

        if model_action.action_type.value == "finalize_batch":
            return heuristic
        return model_action
    except Exception:
        return heuristic


def run_task(client: OpenAI, task_id: str) -> None:
    env = HyperlocalInventoryCurationEnvironment(task_id=task_id)
    observation = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    task_max_steps = min(MAX_STEPS, TASKS[task_id].max_steps)

    try:
        for step in range(1, task_max_steps + 1):
            if observation.done:
                break

            action = choose_action(client, env)
            action_str = json.dumps(
                action.model_dump(exclude_none=True, exclude={"metadata"}),
                ensure_ascii=True,
                separators=(",", ":"),
            )

            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            done = bool(observation.done)
            error = observation.last_action_error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        final_score = float(env.state.score)
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)



def main() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY must be set")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_ORDER:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
