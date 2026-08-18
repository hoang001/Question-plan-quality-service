"""Cấu hình cho question_plan service.

File này đọc `.env`, chuẩn hóa base URL/model/timeout và trả về `AppConfig`
cho luồng đánh giá chất lượng question_plan.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    base_url: str
    api_key: str
    primary_judge_model: str
    fallback_judge_model: str
    use_fallback_judge: bool
    request_timeout_seconds: int
    models_endpoint: str | None
    chat_completions_endpoint: str | None
    gemma_evaluation_concurrency: int
    gemma_request_timeout_seconds: int
    llm_top_p: float
    solution_correctness_model: str = ""
    process_presentation_model: str = ""
    solution_resolver_model: str = ""


def generated_question_reasoning_model(config: AppConfig) -> str:
    """Model reasoning fallback cho generated question."""

    return str(getattr(config, "fallback_judge_model", "") or config.primary_judge_model)


def generated_question_fast_model(config: AppConfig) -> str:
    """Model nhanh cho generated question; giữ mapping Gemma ở PRIMARY_JUDGE_MODEL hiện tại."""

    return str(config.primary_judge_model)


def generated_question_correctness_model(config: AppConfig) -> str:
    """Authoritative model for every mathematical Correctness decision."""

    return str(
        getattr(config, "solution_correctness_model", "")
        or config.primary_judge_model
    )


def generated_question_presentation_model(config: AppConfig) -> str:
    """Small model for process and presentation only."""

    return str(
        getattr(config, "process_presentation_model", "")
        or config.primary_judge_model
    )


def generated_question_resolver_model(config: AppConfig) -> str:
    """Small model for answer extraction/alignment routing."""

    return str(
        getattr(config, "solution_resolver_model", "")
        or config.primary_judge_model
    )


def generated_question_gemma_concurrency(config: AppConfig | None) -> int:
    """Giới hạn tổng số Gemma call đồng thời trong batch."""

    return max(1, min(int(getattr(config, "gemma_evaluation_concurrency", 4)), 4))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc


def load_config(root_dir: Path) -> AppConfig:
    load_dotenv(root_dir / ".env")
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()

    missing = []
    if not base_url:
        missing.append("LLM_BASE_URL")
    if not api_key:
        missing.append("LLM_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Missing required env variable(s): {names}. Create .env from .env.example.")

    primary = os.getenv("PRIMARY_JUDGE_MODEL", "").strip() or "gemma-4-12b-it"
    fallback = os.getenv("FALLBACK_JUDGE_MODEL", "").strip() or "gemma-4-26b"
    correctness = os.getenv("SOLUTION_CORRECTNESS_MODEL", "").strip() or fallback
    presentation = os.getenv("PROCESS_PRESENTATION_MODEL", "").strip() or primary
    resolver = os.getenv("SOLUTION_RESOLVER_MODEL", "").strip() or fallback

    top_p = env_float("LLM_TOP_P", 0.1)
    if not 0 < top_p <= 1:
        raise ConfigError("LLM_TOP_P must be greater than 0 and at most 1.")

    return AppConfig(
        root_dir=root_dir,
        base_url=base_url.rstrip("/") + "/",
        api_key=api_key,
        primary_judge_model=primary,
        fallback_judge_model=fallback,
        use_fallback_judge=env_bool("USE_JUDGE_FALLBACK", True),
        request_timeout_seconds=env_int("REQUEST_TIMEOUT_SECONDS", 60),
        models_endpoint=os.getenv("LLM_MODELS_ENDPOINT", "").strip() or None,
        chat_completions_endpoint=os.getenv("LLM_CHAT_COMPLETIONS_ENDPOINT", "").strip() or None,
        gemma_evaluation_concurrency=env_int("GEMMA_EVALUATION_CONCURRENCY", 4),
        gemma_request_timeout_seconds=env_int("GEMMA_REQUEST_TIMEOUT_SECONDS", 60),
        llm_top_p=top_p,
        solution_correctness_model=correctness,
        process_presentation_model=presentation,
        solution_resolver_model=resolver,
    )
