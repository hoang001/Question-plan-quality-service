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
    model_config = ConfigDict(extra="forbid", strict=True)

    solution_index: int
    order: int
    source_path: str
    source_text: str


class ContextRequirement(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requirement_type: Literal[
        "image",
        "geometry_figure",
        "graph",
        "chart",
        "table",
        "variation_table",
        "number_line",
        "coordinate_plane",
        "diagram",
        "visual_marking",
        "referenced_data",
        "other",
    ]
    description: str = Field(min_length=1)
    availability: Literal["available", "missing", "insufficient"]
    evidence_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_blank_text(self) -> ContextRequirement:
        if not self.description.strip():
            raise ValueError("description must not be blank")
        if not self.evidence_text.strip():
            raise ValueError("evidence_text must not be blank")
        return self


class VisualContextDescription(ContractModel):
    """Grounded transcription/description produced once by the visual Splitter."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_path: str = Field(min_length=1)
    asset_url: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_blank_values(self) -> VisualContextDescription:
        if not all(
            value.strip()
            for value in (self.source_path, self.asset_url, self.description)
        ):
            raise ValueError("visual description fields must not be blank")
        return self


class SolutionSplitOutput(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_requirements: list[ContextRequirement]
    visual_descriptions: list[VisualContextDescription] = Field(default_factory=list)
    states: list[SolutionState]


class CodeTransitionAnalysis(ContractModel):
    """Internal deterministic annotation; never part of the public result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal[
        "verified_valid",
        "verified_invalid",
        "compressed_but_equivalent",
        "unsupported",
        "ambiguous",
        "parse_error",
    ]
    strength: Literal["hard", "soft", "none"]
    transition_type: Literal[
        "equation_transformation",
        "inequality_transformation",
        "expression_transformation",
        "numeric_substitution",
        "numeric_calculation",
        "equality_chain",
        "semantic_reasoning",
        "unsupported",
    ]
    issue_type: Literal[
        "sign_error",
        "calculation_error",
        "invalid_equivalence",
        "inequality_direction_error",
        "division_by_zero",
    ] | None
    operation_count: int | None = Field(default=None, ge=0)
    operation_types: list[str]
    explicit_step_count: int | None = Field(default=None, ge=1)
    max_operations_per_step: int | None = Field(default=None, ge=0)
    total_operation_count: int | None = Field(default=None, ge=0)
    intermediate_states_explicit: bool = False
    failing_pair_index: int | None = Field(default=None, ge=1)
    failing_before: str | None = None
    failing_after: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_strength(self) -> CodeTransitionAnalysis:
        expected_strength = {
            "verified_valid": "hard",
            "verified_invalid": "hard",
            "compressed_but_equivalent": "soft",
            "unsupported": "none",
            "ambiguous": "none",
            "parse_error": "none",
        }[self.status]
        if self.strength != expected_strength:
            raise ValueError("strength không khớp status")
        if self.status == "verified_invalid" and self.issue_type is None:
            raise ValueError("verified_invalid phải có issue_type")
        if self.status != "verified_invalid" and self.issue_type is not None:
            raise ValueError("chỉ verified_invalid được có issue_type")
        return self


class CalculationCheck(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_result: str
    actual_result: str
    matches: bool


class CorrectnessTransitionIssue(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    solution_index: int
    from_order: int | None
    to_order: int
    error_type: Literal[
        "calculation_error",
        "sign_error",
        "coefficient_error",
        "incorrect_transformation",
        "non_equivalent_transformation",
        "logical_error",
        "invalid_theorem_application",
        "domain_error",
        "lost_solution",
        "extraneous_solution",
        "incomplete_reasoning",
        "other_correctness_error",
        "other",
    ]
    evidence_text: str
    calculation_check: CalculationCheck | None


class CorrectnessContextIssue(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    error_type: Literal[
        "insufficient_context",
        "inconsistent_context",
        "contradictory_context",
    ]
    source_path: str
    evidence_text: str


CorrectnessErrorType = Literal[
    "calculation_error",
    "sign_error",
    "coefficient_error",
    "incorrect_transformation",
    "non_equivalent_transformation",
    "logical_error",
    "invalid_theorem_application",
    "domain_error",
    "lost_solution",
    "extraneous_solution",
    "incomplete_reasoning",
    "other_correctness_error",
    "other",
]


class CorrectnessJudgeOutput(ContractModel):
    """Flat schema-constrained semantic decision emitted by Correctness models."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_type: CorrectnessErrorType | None = Field(
        description="Null when status is good; required when status is bad."
    )
    solution_index: int | None = Field(
        ge=0,
        description="Null when status is good; otherwise the anchored solution index.",
    )
    from_order: int | None = Field(
        ge=0,
        description="Null when status is good or for an initial transition.",
    )
    to_order: int | None = Field(
        ge=0,
        description="Null when status is good; required when status is bad.",
    )
    reason: str | None = Field(
        description="Null when status is good; otherwise at most three sentences."
    )
    suggestion: str | None = Field(
        description="Null when status is good; required for bad and at most two sentences."
    )
    status: Literal["good", "bad", "uncertain"] = Field(
        description="Choose after evaluation. When good, code ignores all preceding fields."
    )

    @model_validator(mode="after")
    def validate_status_contract(self) -> CorrectnessJudgeOutput:
        if self.status == "good":
            return self
        if not self.reason or not self.reason.strip():
            raise ValueError("bad/uncertain requires a non-blank reason")
        if self.status == "bad" and (
            self.error_type is None
            or self.solution_index is None
            or self.to_order is None
            or not self.suggestion
            or not self.suggestion.strip()
        ):
            raise ValueError("bad requires error_type, solution_index, to_order and suggestion")
        return self


class CanonicalCorrectnessResult(ContractModel):
    """Code-grounded internal handoff consumed by Gate and Aggregate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    first_invalid_transition: CorrectnessTransitionIssue | None
    context_issue: CorrectnessContextIssue | None
    reason: str
    suggestion: str
    verdict: Literal["good", "bad", "uncertain"]


ProcessPresentationErrorType = Literal[
    "missing_major_step",
    "missing_conclusion",
    "draft_or_self_questioning",
    "unclean_trial_and_error",
    "presentation_self_contradiction",
    "severe_repetition",
    "excessive_verbosity",
    "unsuitable_wording",
]


class ProcessPresentationSemanticOutput(ContractModel):
    """Flat schema-constrained decision emitted by the presentation model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_type: ProcessPresentationErrorType | None
    solution_index: int | None = Field(ge=0)
    state_order: int | None = Field(ge=0)
    reason: str
    suggestion: str
    verdict: Literal["good", "bad", "uncertain"]


class CombinedJudgeCorrectionOutput(ContractModel):
    """Flat one-call correction envelope used only when both Judge contracts fail."""

    model_config = ConfigDict(extra="forbid", strict=True)

    correctness_error_type: CorrectnessErrorType | None
    correctness_solution_index: int | None = Field(ge=0)
    correctness_from_order: int | None = Field(ge=0)
    correctness_to_order: int | None = Field(ge=0)
    correctness_reason: str | None
    correctness_suggestion: str | None
    correctness_status: Literal["good", "bad", "uncertain"]
    process_error_type: ProcessPresentationErrorType | None
    process_solution_index: int | None = Field(ge=0)
    process_state_order: int | None = Field(ge=0)
    process_reason: str
    process_suggestion: str
    process_verdict: Literal["good", "bad", "uncertain"]


class ProcessPresentationIssue(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: Literal["state", "global"]
    solution_index: int
    state_order: int | None
    source_path: str
    evidence_text: str | None
    error_type: ProcessPresentationErrorType


class ProcessPresentationJudgeOutput(ContractModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    issue: ProcessPresentationIssue | None
    reason: str
    suggestion: str
    verdict: Literal["good", "bad", "uncertain"]


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
