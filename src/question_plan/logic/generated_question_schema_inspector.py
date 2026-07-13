"""Inspect schema thực tế của generatedQuestions từ file data."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .generated_question_schema import is_generated_question_object


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def update_keys(bucket: dict[str, Counter], name: str, value: Any) -> None:
    if isinstance(value, dict):
        bucket[name].update(str(key) for key in value.keys())


def inspect_generated_question_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter = Counter()
    keys: dict[str, Counter] = defaultdict(Counter)
    interaction_types: Counter = Counter()
    answer_spec_types: Counter = Counter()
    answer_expected_shapes: Counter = Counter()
    config_keys_by_type: dict[str, Counter] = defaultdict(Counter)
    display_keys_by_type: dict[str, Counter] = defaultdict(Counter)
    content_block_types: Counter = Counter()

    for record in records:
        counts["records"] += 1
        update_keys(keys, "record", record)
        if isinstance(record, dict) and isinstance(record.get("generatedQuestions"), list):
            generated_questions = record.get("generatedQuestions")
        elif is_generated_question_object(record):
            generated_questions = [record]
        else:
            generated_questions = []
        for generated_question in generated_questions if isinstance(generated_questions, list) else []:
            counts["generatedQuestions"] += 1
            update_keys(keys, "generatedQuestion", generated_question)
            if not isinstance(generated_question, dict):
                continue

            for block in generated_question.get("instruction") or []:
                counts["instructionBlocks"] += 1
                update_keys(keys, "contentBlock", block)
                if isinstance(block, dict):
                    content_block_types[str(block.get("type") or "missing")] += 1

            for solution in generated_question.get("solutions") or []:
                counts["solutions"] += 1
                update_keys(keys, "solution", solution)
                if isinstance(solution, dict):
                    for block in solution.get("solutionContent") or []:
                        counts["solutionBlocks"] += 1
                        update_keys(keys, "contentBlock", block)
                        if isinstance(block, dict):
                            content_block_types[str(block.get("type") or "missing")] += 1

            for item in generated_question.get("questionItems") or []:
                counts["questionItems"] += 1
                update_keys(keys, "questionItem", item)
                if not isinstance(item, dict):
                    continue

                for block in item.get("stem") or []:
                    counts["stemBlocks"] += 1
                    update_keys(keys, "contentBlock", block)
                    if isinstance(block, dict):
                        content_block_types[str(block.get("type") or "missing")] += 1

                for hint in item.get("hints") or []:
                    counts["hints"] += 1
                    update_keys(keys, "hint", hint)
                    if isinstance(hint, dict):
                        for block in hint.get("content") or []:
                            counts["hintBlocks"] += 1
                            update_keys(keys, "contentBlock", block)
                            if isinstance(block, dict):
                                content_block_types[str(block.get("type") or "missing")] += 1

                for interaction in item.get("interactions") or []:
                    counts["interactions"] += 1
                    update_keys(keys, "interaction", interaction)
                    if not isinstance(interaction, dict):
                        continue
                    interaction_type = str(interaction.get("type") or "missing")
                    interaction_types[interaction_type] += 1
                    config = interaction.get("config") if isinstance(interaction.get("config"), dict) else {}
                    display = interaction.get("display") if isinstance(interaction.get("display"), dict) else {}
                    config_keys_by_type[interaction_type].update(str(key) for key in config.keys())
                    display_keys_by_type[interaction_type].update(str(key) for key in display.keys())

                for answer_spec in item.get("answerSpecs") or []:
                    counts["answerSpecs"] += 1
                    update_keys(keys, "answerSpec", answer_spec)
                    if not isinstance(answer_spec, dict):
                        continue
                    answer_spec_types[str(answer_spec.get("type") or "missing")] += 1
                    expected = answer_spec.get("expected")
                    if isinstance(expected, list):
                        answer_expected_shapes["list"] += 1
                    elif isinstance(expected, dict):
                        answer_expected_shapes["dict"] += 1
                    else:
                        answer_expected_shapes[type(expected).__name__] += 1

    return {
        "counts": counter_to_dict(counts),
        "keys": {name: sorted(counter.keys()) for name, counter in sorted(keys.items())},
        "interaction_types": counter_to_dict(interaction_types),
        "answer_spec_types": counter_to_dict(answer_spec_types),
        "answer_expected_shapes": counter_to_dict(answer_expected_shapes),
        "content_block_types": counter_to_dict(content_block_types),
        "config_keys_by_type": {
            interaction_type: sorted(counter.keys())
            for interaction_type, counter in sorted(config_keys_by_type.items())
        },
        "display_keys_by_type": {
            interaction_type: sorted(counter.keys())
            for interaction_type, counter in sorted(display_keys_by_type.items())
        },
    }


def format_generated_question_schema_summary(report: dict[str, Any], *, source_name: str = "") -> str:
    lines = ["# GeneratedQuestions Schema Summary", ""]
    if source_name:
        lines.extend([f"Source file: `{source_name}`", ""])

    lines.append("## Counts")
    for key, value in report.get("counts", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Field Keys"])
    for name, keys in report.get("keys", {}).items():
        lines.append(f"- {name}: {', '.join(keys)}")

    lines.extend(["", "## Interaction Types"])
    for interaction_type, count in report.get("interaction_types", {}).items():
        lines.append(f"- {interaction_type}: {count}")

    lines.extend(["", "## AnswerSpec Types"])
    for answer_spec_type, count in report.get("answer_spec_types", {}).items():
        lines.append(f"- {answer_spec_type}: {count}")

    lines.extend(["", "## Answer Expected Shapes"])
    for shape, count in report.get("answer_expected_shapes", {}).items():
        lines.append(f"- {shape}: {count}")

    lines.extend(["", "## Content Block Types"])
    for block_type, count in report.get("content_block_types", {}).items():
        lines.append(f"- {block_type}: {count}")

    lines.extend(["", "## Config Keys By Interaction Type"])
    for interaction_type, keys in report.get("config_keys_by_type", {}).items():
        lines.append(f"- {interaction_type}: {', '.join(keys)}")

    lines.extend(["", "## Display Keys By Interaction Type"])
    for interaction_type, keys in report.get("display_keys_by_type", {}).items():
        lines.append(f"- {interaction_type}: {', '.join(keys)}")

    lines.append("")
    return "\n".join(lines)
