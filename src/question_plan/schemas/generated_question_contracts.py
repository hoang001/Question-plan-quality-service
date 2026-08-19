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


class DirectSolutionCorrectnessComment(ContractModel):
    """One mathematical comment found while reading an original solution."""

    model_config = ConfigDict(extra="forbid", strict=True)

    solution_index: int = Field(ge=0)
    evidence: str
    reason: str


class DirectSolutionOpeningCheck(ContractModel):
    """Objective comparison between the problem and a solution's first claim."""

    model_config = ConfigDict(extra="forbid", strict=True)

    solution_index: int = Field(ge=0)
    evidence: str
    analysis: str


class DirectSolutionCorrectnessOutput(ContractModel):
    """Correctness observations before the independent review call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    opening_checks: list[DirectSolutionOpeningCheck] = Field(min_length=1)
    comments: list[DirectSolutionCorrectnessComment]


ProcessPresentationErrorType = Literal[
    "missing_major_step",
    "missing_conclusion",
    "draft_or_self_questioning",
    "unclean_trial_and_error",
    "presentation_self_contradiction",
    "severe_repetition",
    "excessive_verbosity",
    "redundant_step",
    "unsuitable_wording",
]


class ProcessPresentationComment(ContractModel):
    """One process/presentation comment before independent review."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_type: ProcessPresentationErrorType | None
    solution_index: int = Field(ge=0)
    evidence: str
    reason: str
    suggestion: str


class ProcessPresentationSemanticOutput(ContractModel):
    """Process/presentation observations before the independent review call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    comments: list[ProcessPresentationComment]


CommentDisposition = Literal["blocking", "advisory", "rejected"]


class ReviewedJudgeComment(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    comment_id: str
    disposition: CommentDisposition
    review_reason: str = Field(min_length=1, max_length=400)
    review_suggestion: str = Field(max_length=300)


class JudgeCommentsReviewOutput(ContractModel):
    """Review dispositions for the comments produced by the two judges."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reviewed_comments: list[ReviewedJudgeComment]


class FinalAnswer(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    matched_option_id: str | None
    correctOptionIds: list[str]
    expected: Any
    evidence_from_solution: Any = None


class ResolverFieldFix(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    value: Any
    reason: str
    suggestion: str


class ResolverIssue(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
    model_config = ConfigDict(extra="forbid", strict=True)

    resolver_status: Literal["resolved", "needs_manual_review"]
    final_answer: FinalAnswer
    answerSpec_matches_solution: bool
    answer_spec_alignment: Literal["matched", "equivalent", "mismatched"] | None = None
    fields_to_fix: list[ResolverFieldFix]
    issues: list[ResolverIssue]


class JsonPatch(ContractModel):
    op: Literal["replace"]
    path: str
    value: Any
    reason: str = Field(min_length=1, max_length=500)


class RepairPatchOutput(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patches: list[JsonPatch] = Field(min_length=1)


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
