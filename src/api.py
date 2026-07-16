"""FastAPI wrapper cho question_plan service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query

from .question_plan.flows.generated_question_service import evaluate_generated_questions
from .question_plan.flows.service import evaluate_question_plan, evaluate_question_plans


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="Question Plan Quality Service")


SOURCE_RECORD_EXAMPLE: dict[str, Any] = {
    "_id": "demo-record-001",
    "name": "Demo question plan",
    "question": "Giải phương trình 2x + 3 = 7.",
    "images": [],
    "answer": "Ta có 2x + 3 = 7 nên 2x = 4, suy ra x = 2.",
    "answer_images": [],
    "question_plan": {
        "type": "advanced_question_plan",
        "plan": [
            {
                "questionOrder": 1,
                "questionStatement": "Giải phương trình 2x + 3 = 7.",
                "questionItems": [
                    {
                        "itemOrder": 1,
                        "requirement": "Tìm nghiệm của phương trình.",
                        "interactions": [
                            {
                                "interactionOrder": 1,
                                "interactionType": "short_answer",
                                "interactionRequirement": "Giá trị của x",
                            }
                        ],
                    }
                ],
            }
        ],
    },
    "start_page": None,
    "end_page": None,
}


GENERATED_QUESTION_OBJECT_EXAMPLE: dict[str, Any] = {
    "_id": "generated-demo-001",
    "aiId": "demo-ai-flow",
    "difficulty": "easy",
    "bloom": "apply",
    "interactionTypes": ["short_answer"],
    "instruction": [{"id": "intro", "type": "text", "text": "Giải phương trình 2x + 3 = 7."}],
    "questionItems": [
        {
            "id": "item-demo-001",
            "stem": [{"id": "stem", "type": "text", "text": "Nhập giá trị của x."}],
            "interactions": [
                {
                    "id": "x_value",
                    "type": "short_answer",
                    "config": {"inputMode": "numeric"},
                    "display": {"layout": "auto"},
                }
            ],
            "answerSpecs": [
                {
                    "interactionId": "x_value",
                    "type": "short_answer",
                    "expected": [
                        {
                            "inputMode": "numeric",
                            "value": {"correctValue": 2, "acceptableValues": []},
                            "equivalence": {"type": "numeric_equivalence"},
                        }
                    ],
                }
            ],
        }
    ],
    "solutions": [
        {
            "solverName": "default",
            "solutionContent": [{"id": "solution", "type": "text", "text": "Ta có x = 2."}],
        }
    ],
}


GENERATED_QUESTION_WRAPPER_EXAMPLE: dict[str, Any] = {
    "_id": "source-record-demo",
    "name": "Wrapper demo",
    "generatedQuestions": [GENERATED_QUESTION_OBJECT_EXAMPLE],
}


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected_api_key = os.getenv("QUESTION_PLAN_API_KEY", "").strip()
    if expected_api_key and x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai X-API-Key.")


@app.get("/health")
def health(_: None = Depends(verify_api_key)) -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate-question-plan")
def evaluate_question_plan_api(
    record: dict[str, Any] = Body(
        ...,
        description="Một source record object gồm question, answer và question_plan.",
        examples=[SOURCE_RECORD_EXAMPLE],
    ),
    is_loop: bool = Query(default=False, description="Bật loop/refinement sau repair."),
    max_loop: int = Query(default=3, description="Số vòng loop tối đa, service sẽ clamp trong khoảng 1..3."),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        return evaluate_question_plan(record, is_loop=is_loop, max_loop=max_loop)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đánh giá question_plan: {exc}") from exc


@app.post("/evaluate-question-plans")
def evaluate_question_plans_api(
    records: list[dict[str, Any]] = Body(
        ...,
        description="Danh sách source record object.",
        examples=[[SOURCE_RECORD_EXAMPLE]],
    ),
    is_loop: bool = Query(default=False, description="Bật loop/refinement sau repair cho từng record."),
    max_loop: int = Query(default=3, description="Số vòng loop tối đa mỗi record, service sẽ clamp trong khoảng 1..3."),
    _: None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    try:
        return evaluate_question_plans(records, is_loop=is_loop, max_loop=max_loop)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đánh giá danh sách question_plan: {exc}") from exc


@app.post("/evaluate-generated-questions")
def evaluate_generated_questions_api(
    payload: Any = Body(
        ...,
        description=(
            "Một generated question object, list generated question object, "
            "hoặc wrapper cũ có field generatedQuestions."
        ),
        examples=[GENERATED_QUESTION_OBJECT_EXAMPLE, [GENERATED_QUESTION_OBJECT_EXAMPLE], GENERATED_QUESTION_WRAPPER_EXAMPLE],
    ),
    strict_mode: bool = Query(default=True, description="Nếu true, needs_review/bad làm is_good=false."),
    debug: bool = Query(default=True, description="false trả compact output; true trả diagnostics an toàn."),
    auto_repair: bool = Query(default=True, description="Bật repair tự động nếu lỗi sửa được an toàn."),
    max_loop: int = Query(default=1, description="Số vòng repair/check tối đa, service clamp trong khoảng 1..3."),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        return evaluate_generated_questions(
            payload,
            strict_mode=strict_mode,
            debug=debug,
            auto_repair=auto_repair,
            max_loop=max_loop,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đánh giá generatedQuestions: {exc}") from exc
