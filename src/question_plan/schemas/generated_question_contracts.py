"""Single source of truth for LLM output contracts in generated-question flow."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for name, field in cls.model_fields.items():
            if normalized.get(name) is not None:
                continue
            if field.default_factory is not None:
                normalized[name] = field.default_factory()
            elif not field.is_required() and field.default is not None:
                normalized[name] = field.default
        return normalized


class SolutionState(ContractModel):
    solution_index: int = 0
    order: int
    source_path: str
    source_text: str


class SolutionSplitOutput(ContractModel):
    states: list[SolutionState] = Field(default_factory=list)


class StageJudgeResult(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    solution_index: int = 0
    stage: int
    from_order: int
    to_order: int
    kiem_tra_so_hoc: str | None
    kiem_tra_lap_luan: str
    trang_thai_buoc: bool
    reason: str


class TransitionJudgeOutput(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stage_results: list[StageJudgeResult]
    reason: str
    suggestion: str
    verdict: Literal["good", "bad", "uncertain"]


class FinalAnswer(ContractModel):
    text: str | None = None
    matched_option_id: str | None = None
    correctOptionIds: list[str] = Field(default_factory=list)
    expected: Any = None
    evidence_from_solution: str


class ResolverFieldFix(ContractModel):
    path: str
    value: Any
    reason: str = ""
    suggestion: str = ""


class ResolverIssue(ContractModel):
    severity: Literal["warning", "needs_review", "bad"]
    category: Literal["solution_anchor_consistency", "solution_quality", "hint_quality"]
    location: str
    reason: str
    suggestion: str
    repair_intent: Literal[
        "align_fields_to_solution",
        "align_hint_to_solution",
        "clean_solution_reasoning",
        "needs_manual_review",
    ]


class SolutionResolverOutput(ContractModel):
    resolver_status: Literal["resolved", "needs_manual_review"]
    final_answer: FinalAnswer
    answerSpec_matches_solution: bool
    fields_to_fix: list[ResolverFieldFix] = Field(default_factory=list)
    issues: list[ResolverIssue] = Field(default_factory=list)


class JsonPatch(ContractModel):
    op: Literal["replace", "add", "remove"]
    path: str
    value: Any = None


class ScopedRepairOutput(ContractModel):
    repair_status: Literal["repaired", "failed", "needs_manual_review"]
    patches: list[JsonPatch] = Field(default_factory=list)
    failed_reason: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class SpellingIssue(ContractModel):
    severity: Literal["warning", "needs_review", "bad"]
    category: Literal["spelling_wording"] = "spelling_wording"
    location: str
    reason: str
    suggestion: str
    error_snippet: str = ""
    corrected_text: str | None = None
    repair_intent: Literal["fix_spelling_wording"] = "fix_spelling_wording"


class SpellingOutput(ContractModel):
    issues: list[SpellingIssue] = Field(default_factory=list)


def validation_error_text(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc") or [])
    message = {
        "missing": "Thiếu trường bắt buộc",
        "model_type": "Phải là một đối tượng JSON hợp lệ",
        "list_type": "Phải là một danh sách hợp lệ",
        "string_type": "Phải là một chuỗi hợp lệ",
        "bool_type": "Phải là giá trị đúng hoặc sai hợp lệ",
        "literal_error": "Giá trị không thuộc tập giá trị được cho phép",
    }.get(str(error.get("type") or ""), "Dữ liệu không đúng kiểu hoặc cấu trúc yêu cầu")
    return f"{location}: {message}" if location else message


def contract_schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)
