"""Narrow deterministic analysis for canonical solution transitions.

The analyzer annotates every transition before the Correctness Judge runs.  It
never creates a public issue and deliberately returns ``unsupported`` when a
claim is outside the small, safe symbolic subset implemented here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from ..schemas.generated_question_contracts import CodeTransitionAnalysis


_RELATION = re.compile(r"(<=|>=|!=|=|<|>)")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_UNSAFE_DOMAIN = re.compile(
    r"(?:\\(?:sqrt|log|ln|sin|cos|tan|int)\b|∫|"
    r"\b(?:sqrt|log|ln|sin|cos|tan)\b|\bcăn\b|\blog\s+cơ\s+số\b|"
    r"\b(?:nguyên\s+hàm|tích\s+phân|đạo\s+hàm)\b|[A-Za-z]\s*['′])",
    re.IGNORECASE,
)
_VARIABLE_DIVISION = re.compile(
    r"\bchia\b.{0,80}\bcho\s+x\b|\bchia\b.{0,80}\bcho\s*\([^)]*x",
    re.IGNORECASE,
)
_DIVISION_BY_ZERO = re.compile(r"\bchia\b.{0,80}\bcho\s+0\b", re.IGNORECASE)
_LATEX_FRACTION = re.compile(r"\\(?:dfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_NUMERIC_LITERAL = r"[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)"
_STANDALONE_NUMERIC_ASSERTION = re.compile(
    rf"""
    ^\s*
    (?:(?:
        Ta\s+có|Suy\s+ra|Do\s+đó|Vậy|Khi\s+đó|Tính\s+được|Kết\s+quả\s+là|
        Theo\s+bảng\s+(?:nhân|chia)\s+\d+\s*,
    )\s*:?\s*)?
    \$?\s*
    (?P<expression>
        (?P<left>{_NUMERIC_LITERAL})\s*
        (?P<operator>[+\-*/×·÷])\s*
        (?P<right>{_NUMERIC_LITERAL})\s*
        =\s*(?P<actual>{_NUMERIC_LITERAL})
    )
    \s*\$?\s*[.]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXPLICIT_STEP_BOUNDARY = re.compile(
    r"(?:[\r\n]+|[.;](?!\d)|,\s*(?=(?:sau\s+đó|tiếp\s+theo)\b))",
    re.IGNORECASE,
)
_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "*",
        "·": "*",
        "÷": "/",
        "≤": "<=",
        "≥": ">=",
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "[": "(",
        "]": ")",
        "\u00a0": " ",
    }
)


class _Unsupported(ValueError):
    pass


class _Ambiguous(ValueError):
    pass


class _ParseError(ValueError):
    pass


@dataclass(frozen=True)
class _Claim:
    raw: str
    normalized: str
    parts: tuple[Any, ...]
    relations: tuple[str, ...]
    symbols: tuple[Any, ...]
    start: int
    end: int


@dataclass(frozen=True)
class _MandatoryArithmeticFinding:
    evidence_text: str
    expected_result: str
    actual_result: str
    analysis: dict[str, Any]


def _analysis(
    status: str,
    transition_type: str,
    *,
    issue_type: str | None = None,
    operation_count: int | None = None,
    operation_types: list[str] | None = None,
    explicit_step_count: int | None = None,
    max_operations_per_step: int | None = None,
    total_operation_count: int | None = None,
    intermediate_states_explicit: bool = False,
    failing_pair_index: int | None = None,
    failing_before: str | None = None,
    failing_after: str | None = None,
    reason: str,
) -> dict[str, Any]:
    strength = {
        "verified_valid": "hard",
        "verified_invalid": "hard",
        "compressed_but_equivalent": "soft",
        "unsupported": "none",
        "ambiguous": "none",
        "parse_error": "none",
    }[status]
    payload = {
        "status": status,
        "strength": strength,
        "transition_type": transition_type,
        "issue_type": issue_type,
        "operation_count": operation_count,
        "operation_types": operation_types or [],
        "explicit_step_count": (
            explicit_step_count
            if explicit_step_count is not None
            else 1 if operation_count is not None else None
        ),
        "max_operations_per_step": (
            max_operations_per_step
            if max_operations_per_step is not None
            else operation_count
        ),
        "total_operation_count": (
            total_operation_count
            if total_operation_count is not None
            else operation_count
        ),
        "intermediate_states_explicit": intermediate_states_explicit,
        "failing_pair_index": failing_pair_index,
        "failing_before": failing_before,
        "failing_after": failing_after,
        "reason": reason,
    }
    return CodeTransitionAnalysis.model_validate(payload).model_dump()


def _normalize_math(value: str) -> str:
    text = unicodedata.normalize("NFC", value).translate(_TRANSLATION)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_FRACTION.sub(r"((\1)/(\2))", text)
    text = text.replace("{", "(").replace("}", ")").replace("^", "**")
    text = re.sub(r"(?<=[0-9)])\s*(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z])\s*(?=\()", "*", text)
    return re.sub(r"\s+", "", text.strip(" \t\r\n,;:.!?\"'"))


def _fraction_literal(value: str) -> Fraction:
    return Fraction(value.replace(",", "."))


def _format_fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _standalone_numeric_assertion(
    value: str,
) -> _MandatoryArithmeticFinding | None:
    """Chỉ hard-check một mệnh đề số học độc lập khớp toàn bộ state text."""

    match = _STANDALONE_NUMERIC_ASSERTION.fullmatch(str(value or ""))
    if match is None:
        return None
    left = _fraction_literal(match.group("left"))
    right = _fraction_literal(match.group("right"))
    operator = match.group("operator")
    if operator == "+":
        expected = left + right
    elif operator == "-":
        expected = left - right
    elif operator in {"*", "×", "·"}:
        expected = left * right
    else:
        if right == 0:
            return _MandatoryArithmeticFinding(
                evidence_text=match.group("expression"),
                expected_result="không xác định",
                actual_result=match.group("actual"),
                analysis=_analysis(
                    "verified_invalid",
                    "numeric_calculation",
                    issue_type="division_by_zero",
                    operation_count=1,
                    operation_types=["evaluate_single_numeric_operation"],
                    reason="Mệnh đề số học độc lập thực hiện phép chia cho 0.",
                ),
            )
        expected = left / right
    actual = _fraction_literal(match.group("actual"))
    if expected == actual:
        return _MandatoryArithmeticFinding(
            evidence_text=match.group("expression"),
            expected_result=_format_fraction(expected),
            actual_result=match.group("actual"),
            analysis=_analysis(
                "verified_valid",
                "numeric_calculation",
                operation_count=1,
                operation_types=["evaluate_single_numeric_operation"],
                reason=(
                    "Mệnh đề số học độc lập có đúng một phép tính và kết quả "
                    "khớp kết quả code tính chính xác."
                ),
            ),
        )
    return _MandatoryArithmeticFinding(
        evidence_text=match.group("expression"),
        expected_result=_format_fraction(expected),
        actual_result=match.group("actual"),
        analysis=_analysis(
            "verified_invalid",
            "numeric_calculation",
            issue_type="calculation_error",
            operation_count=1,
            operation_types=["evaluate_single_numeric_operation"],
            reason=(
                "Mệnh đề số học độc lập có đúng một phép tính và kết quả viết ra "
                "không khớp kết quả code tính chính xác."
            ),
        ),
    )


def _parse_expression(raw: str, variable_names: set[str]) -> Any:
    try:
        from sympy import Symbol, count_ops
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError as exc:  # pragma: no cover
        raise _Unsupported("Thiếu thư viện SymPy.") from exc

    try:
        expression = parse_expr(
            raw,
            local_dict={
                name: Symbol(name, real=True)
                for name in variable_names
            },
            transformations=standard_transformations
            + (implicit_multiplication_application, convert_xor),
            evaluate=True,
        )
    except Exception as exc:
        raise _ParseError("Không parse được biểu thức.") from exc
    if int(count_ops(expression)) > 48:
        raise _Unsupported("Biểu thức vượt phạm vi phức tạp an toàn.")
    return expression


def _parse_candidate(raw: str, start: int, end: int) -> _Claim:
    normalized = _normalize_math(raw)
    if not normalized or len(normalized) > 240:
        raise _ParseError("Math claim rỗng hoặc quá dài.")
    if _UNSAFE_DOMAIN.search(normalized):
        raise _Unsupported("Claim chứa căn, log hoặc hàm ngoài phạm vi an toàn.")
    relations = tuple(_RELATION.findall(normalized))
    raw_parts = tuple(_RELATION.split(normalized)[::2])
    if any(not part for part in raw_parts) or len(raw_parts) != len(relations) + 1:
        raise _ParseError("Quan hệ toán học không hoàn chỉnh.")
    identifiers = set(_IDENTIFIER.findall(normalized))
    if any(len(identifier) != 1 for identifier in identifiers) or len(identifiers) > 1:
        raise _Unsupported("Claim không thuộc phạm vi một biến đơn giản.")
    if any("/" in part and identifiers and re.search(r"/[^=<>]*[A-Za-z]", part) for part in raw_parts):
        raise _Unsupported("Phân thức chứa biến nằm ngoài phạm vi an toàn.")
    expressions = tuple(_parse_expression(part, identifiers) for part in raw_parts)
    symbols = tuple(
        sorted(set().union(*(expression.free_symbols for expression in expressions)), key=str)
    )
    if len(symbols) > 1:
        raise _Unsupported("Claim có nhiều hơn một biến.")
    return _Claim(raw, normalized, expressions, relations, symbols, start, end)


def _extract_claim(
    value: str,
    *,
    allow_numeric_fragment: bool = False,
) -> _Claim:
    text = str(value or "").replace("\n", " ")
    tokens = list(re.finditer(r"\S+", text))
    candidates: dict[str, _Claim] = {}
    failed_relational_candidates: list[tuple[int, int, int]] = []
    saw_unsupported = False
    for start_index in range(len(tokens)):
        for end_index in range(start_index + 1, len(tokens) + 1):
            start = tokens[start_index].start()
            end = tokens[end_index - 1].end()
            raw = text[start:end].strip(" \t\r\n,;:.!?\"'")
            if not raw or not (
                _RELATION.search(raw)
                or any(operator in raw for operator in ("+", "-", "*", "/", "^", "×", "÷"))
            ):
                continue
            try:
                claim = _parse_candidate(raw, start, end)
            except _Unsupported:
                saw_unsupported = True
                relation_count = len(_RELATION.findall(_normalize_math(raw)))
                if relation_count:
                    failed_relational_candidates.append((start, end, relation_count))
                continue
            except _ParseError:
                relation_count = len(_RELATION.findall(_normalize_math(raw)))
                if relation_count:
                    failed_relational_candidates.append((start, end, relation_count))
                continue
            if not claim.relations and not claim.symbols:
                continue
            candidates[claim.normalized] = claim
    if not candidates:
        if saw_unsupported or _RELATION.search(text):
            raise _Unsupported("Không có math claim đơn giản, chắc chắn để code kiểm tra.")
        raise _Unsupported("Transition là suy luận ngôn ngữ hoặc ngoài phạm vi code.")

    values = list(candidates.values())
    maximal = [
        claim
        for claim in values
        if not any(
            other.start <= claim.start
            and other.end >= claim.end
            and (other.start < claim.start or other.end > claim.end)
            and len(other.relations) >= len(claim.relations)
            for other in values
        )
    ]
    ranked = sorted(
        maximal,
        key=lambda claim: (
            len(claim.relations),
            claim.start,
            len(claim.normalized),
        ),
        reverse=True,
    )
    best = ranked[0]
    if any(
        failed_start <= best.start
        and failed_end >= best.end
        and failed_relation_count > len(best.relations)
        for failed_start, failed_end, failed_relation_count in failed_relational_candidates
    ) and not (
        allow_numeric_fragment
        and best.relations == ("=",)
        and not best.symbols
    ):
        raise _Unsupported(
            "Math claim đầy đủ nằm ngoài verifier; không hard-verify fragment làm mất quan hệ."
        )
    if len(ranked) > 1:
        first_score = (len(best.relations), len(best.normalized), best.start)
        second = ranked[1]
        second_score = (len(second.relations), len(second.normalized), second.start)
        if first_score == second_score and best.normalized != second.normalized:
            raise _Ambiguous("Có nhiều math claim ngang hạng.")
    return best


def _extract_explicit_ordered_claims(value: str) -> list[_Claim]:
    """Extract one unambiguous claim from each explicit text step.

    Sentence, newline and semicolon boundaries are intentionally required.
    A comma is accepted only with an explicit ordered connector such as
    ``sau đó`` or ``tiếp theo``; other commas may introduce alternatives.
    """

    text = str(value or "")
    segments = [
        segment.strip()
        for segment in _EXPLICIT_STEP_BOUNDARY.split(text)
        if segment.strip()
    ]
    if len(segments) < 2:
        return []

    claims: list[_Claim] = []
    for segment in segments:
        try:
            claim = _extract_claim(segment)
        except (_Unsupported, _Ambiguous, _ParseError):
            continue
        if claims and claim.normalized == claims[-1].normalized:
            continue
        claims.append(claim)
    return claims if len(claims) >= 2 else []


def _analyze_explicit_claim_chain(before: str, after: str) -> dict[str, Any] | None:
    """Verify a conservative chain of explicitly written intermediate states."""

    after_claims = _extract_explicit_ordered_claims(after)
    if len(after_claims) < 2:
        return None
    try:
        before_claim = _extract_claim(before)
    except (_Unsupported, _Ambiguous, _ParseError):
        return None

    claims = [before_claim, *after_claims]
    relation_shapes = {claim.relations for claim in claims}
    symbol_shapes = {
        tuple(str(symbol) for symbol in claim.symbols) for claim in claims
    }
    if relation_shapes != {("=",)} or len(symbol_shapes) != 1:
        return None

    pair_results: list[dict[str, Any]] = []
    for pair_index, (previous, current) in enumerate(
        zip(claims, claims[1:]),
        start=1,
    ):
        pair = analyze_code_transition(before=previous.raw, after=current.raw)
        if pair.get("status") == "verified_invalid" and pair.get("strength") == "hard":
            return _analysis(
                "verified_invalid",
                str(pair.get("transition_type") or "equation_transformation"),
                issue_type=str(pair.get("issue_type") or "invalid_equivalence"),
                operation_count=pair.get("operation_count"),
                operation_types=list(pair.get("operation_types") or []),
                explicit_step_count=len(after_claims),
                max_operations_per_step=pair.get("operation_count"),
                total_operation_count=None,
                intermediate_states_explicit=True,
                failing_pair_index=pair_index,
                failing_before=previous.raw,
                failing_after=current.raw,
                reason=(
                    f"Chuỗi có trạng thái trung gian tường minh nhưng cặp bước {pair_index} "
                    f"không hợp lệ: {pair.get('reason') or 'không bảo toàn quan hệ.'}"
                ),
            )
        if pair.get("status") != "verified_valid" or pair.get("strength") != "hard":
            return None
        count = pair.get("operation_count")
        if not isinstance(count, int) or count != 1:
            return None
        pair_results.append(pair)

    operation_types = [
        operation_type
        for pair in pair_results
        for operation_type in pair.get("operation_types") or []
    ]
    total_operation_count = sum(
        int(pair.get("operation_count") or 0) for pair in pair_results
    )
    return _analysis(
        "verified_valid",
        "equation_transformation",
        operation_count=1,
        operation_types=operation_types,
        explicit_step_count=len(pair_results),
        max_operations_per_step=1,
        total_operation_count=total_operation_count,
        intermediate_states_explicit=True,
        reason=(
            "Code xác minh chuỗi trạng thái trung gian được viết tường minh; "
            "mỗi cặp trạng thái liền kề chỉ có một phép biến đổi cơ bản."
        ),
    )


def _simple_assignment_value(claim: _Claim, variable: Any) -> Any | None:
    from sympy import simplify

    if len(claim.relations) != 1 or claim.relations[0] != "=":
        return None
    if simplify(claim.parts[0] - variable) == 0 and not claim.parts[1].free_symbols:
        return claim.parts[1]
    if simplify(claim.parts[1] - variable) == 0 and not claim.parts[0].free_symbols:
        return claim.parts[0]
    return None


def _odd_monomial_assignment_equivalent(
    polynomial_claim: _Claim,
    assignment_claim: _Claim,
    variable: Any,
) -> bool:
    """Prove `a*x^(2k+1)+b=0 <=> x=c` over the real numbers."""

    from sympy import Poly, simplify

    candidate = _simple_assignment_value(assignment_claim, variable)
    if candidate is None or candidate.free_symbols:
        return False
    polynomial = Poly(
        simplify(polynomial_claim.parts[0] - polynomial_claim.parts[-1]),
        variable,
    )
    degree = int(polynomial.degree())
    if degree <= 1 or degree % 2 == 0:
        return False
    if any(monomial not in {(degree,), (0,)} for monomial in polynomial.monoms()):
        return False
    return bool(simplify(polynomial.as_expr().subs(variable, candidate)) == 0)


def _equation_relation(before: _Claim, after: _Claim) -> bool:
    from sympy import Poly, simplify

    before_zero = simplify(before.parts[0] - before.parts[-1])
    after_zero = simplify(after.parts[0] - after.parts[-1])
    symbols = tuple(sorted(set(before.symbols) | set(after.symbols), key=str))
    if not symbols:
        return bool(before_zero == 0) == bool(after_zero == 0)
    variable = symbols[0]
    before_poly = Poly(before_zero, variable)
    after_poly = Poly(after_zero, variable)
    if before_poly.is_zero or after_poly.is_zero:
        return before_poly.is_zero and after_poly.is_zero
    monomials = set(before_poly.monoms()) | set(after_poly.monoms())
    coefficient_pairs = [
        (
            before_poly.coeff_monomial(monomial),
            after_poly.coeff_monomial(monomial),
        )
        for monomial in monomials
    ]
    ratios = [
        simplify(right / left)
        for left, right in coefficient_pairs
        if left != 0 and right != 0
    ]
    same_polynomial_relation = bool(
        len(ratios) == len(coefficient_pairs)
        and ratios
        and all(ratio == ratios[0] for ratio in ratios)
        and ratios[0].is_number
        and ratios[0] != 0
    )
    if same_polynomial_relation:
        return True
    return _odd_monomial_assignment_equivalent(before, after, variable) or (
        _odd_monomial_assignment_equivalent(after, before, variable)
    )


def _ensure_supported_polynomial_equations(*claims: _Claim) -> None:
    from sympy import Poly, simplify

    symbols = tuple(
        sorted(set().union(*(set(claim.symbols) for claim in claims)), key=str)
    )
    if len(symbols) != 1:
        raise _Unsupported("Chỉ hard-verify phương trình đa thức một ẩn.")
    variable = symbols[0]
    nonlinear_degrees: set[int] = set()
    for claim in claims:
        if not claim.relations or any(relation != "=" for relation in claim.relations):
            continue
        polynomial = Poly(simplify(claim.parts[0] - claim.parts[-1]), variable)
        degree = int(polynomial.degree())
        if degree > 6:
            raise _Unsupported("Bậc đa thức vượt phạm vi hard-verify an toàn.")
        if degree <= 1:
            continue
        if degree % 2 == 0 or any(
            monomial not in {(degree,), (0,)} for monomial in polynomial.monoms()
        ):
            raise _Unsupported(
                "Chỉ hard-verify phi tuyến dạng đơn thức bậc lẻ a*x^n = b."
            )
        nonlinear_degrees.add(degree)
    if len(nonlinear_degrees) > 1:
        raise _Unsupported("Không so sánh hai lũy thừa phi tuyến khác bậc.")
    if nonlinear_degrees:
        for claim in claims:
            polynomial = Poly(
                simplify(claim.parts[0] - claim.parts[-1]),
                variable,
            )
            if int(polynomial.degree()) == 1 and (
                _simple_assignment_value(claim, variable) is None
            ):
                raise _Unsupported(
                    "Chỉ so sánh đơn thức bậc lẻ với một assignment nghiệm rõ ràng."
                )


def _numeric_chain_valid(claim: _Claim) -> bool:
    from sympy import simplify

    for left, right in zip(claim.parts, claim.parts[1:]):
        if not left.free_symbols and not right.free_symbols and simplify(left - right) != 0:
            return False
    return True


def _same_nonzero_ratio(new_value: Any, old_value: Any) -> Any | None:
    from sympy import simplify

    if simplify(old_value) == 0:
        return None
    ratio = simplify(new_value / old_value)
    return ratio if ratio.is_number and ratio != 0 else None


def _monomial_equation_shape(claim: _Claim, variable: Any) -> tuple[int, Any] | None:
    """Return `(degree, coefficient)` for `a*x^n = constant` in either orientation."""

    from sympy import Poly, simplify

    if len(claim.relations) != 1 or claim.relations[0] != "=":
        return None
    for variable_side, constant_side in (
        (claim.parts[0], claim.parts[1]),
        (claim.parts[1], claim.parts[0]),
    ):
        if constant_side.free_symbols:
            continue
        polynomial = Poly(simplify(variable_side), variable)
        degree = int(polynomial.degree())
        if degree < 1 or polynomial.monoms() != [(degree,)]:
            continue
        coefficient = polynomial.coeff_monomial(variable**degree)
        if coefficient != 0:
            return degree, coefficient
    return None


def _operation_count(before: _Claim, after: _Claim) -> tuple[int, list[str]] | None:
    from sympy import Poly, simplify

    if len(before.parts) < 2 or len(after.parts) < 2:
        return (1, ["simplify_expression"])
    left_delta = simplify(after.parts[0] - before.parts[0])
    right_delta = simplify(after.parts[-1] - before.parts[-1])
    if left_delta == right_delta and left_delta != 0:
        return (
            1,
            [
                "subtract_same_value_both_sides"
                if left_delta.could_extract_minus_sign()
                else "add_same_value_both_sides"
            ],
        )
    left_ratio = _same_nonzero_ratio(after.parts[0], before.parts[0])
    right_ratio = _same_nonzero_ratio(after.parts[-1], before.parts[-1])
    if left_ratio is not None and left_ratio == right_ratio:
        return (
            1,
            [
                "divide_both_sides"
                if abs(left_ratio) < 1
                else "multiply_both_sides"
            ],
        )

    symbols = tuple(sorted(set(before.symbols) | set(after.symbols), key=str))
    if len(symbols) != 1:
        return None
    variable = symbols[0]
    target_solved = (
        simplify(after.parts[0] - variable) == 0
        and not after.parts[-1].free_symbols
    ) or (
        simplify(after.parts[-1] - variable) == 0
        and not after.parts[0].free_symbols
    )
    if not target_solved:
        return None
    polynomial = Poly(simplify(before.parts[0] - before.parts[-1]), variable)
    degree = int(polynomial.degree())
    monomial_shape = _monomial_equation_shape(before, variable)
    if monomial_shape is not None:
        monomial_degree, coefficient = monomial_shape
        operations: list[str] = []
        if coefficient != 1:
            operations.append("divide_both_sides")
        if monomial_degree > 1:
            if monomial_degree % 2 == 0:
                raise _Unsupported("Căn bậc chẵn cần kiểm tra đủ hai nhánh nghiệm.")
            operations.append("take_odd_root_both_sides")
        return len(operations), operations
    if degree != 1:
        raise _Unsupported("Đa thức phi tuyến không ở dạng đơn thức giải được an toàn.")
    coefficient = polynomial.coeff_monomial(variable)
    constant = polynomial.coeff_monomial(1)
    operations: list[str] = []
    if constant != 0:
        operations.append("add_or_subtract_same_value_both_sides")
    if coefficient not in {1, -1}:
        operations.append("divide_both_sides")
    return len(operations), operations


def _looks_like_sign_error(before: _Claim, after: _Claim) -> bool:
    from sympy import Poly, simplify

    symbols = tuple(sorted(set(before.symbols) | set(after.symbols), key=str))
    if len(symbols) != 1:
        return False
    variable = symbols[0]
    try:
        before_left = Poly(before.parts[0], variable)
        if before_left.degree() != 1 or after.parts[0].free_symbols != {variable}:
            return False
        coefficient = before_left.coeff_monomial(variable)
        constant = before_left.coeff_monomial(1)
        return bool(
            constant != 0
            and simplify(after.parts[0] - coefficient * variable) == 0
            and simplify(after.parts[-1] - (before.parts[-1] + constant)) == 0
        )
    except Exception:
        return False


def _inequality_equivalent(before: _Claim, after: _Claim) -> bool:
    if len(before.relations) != 1 or len(after.relations) != 1:
        raise _Unsupported("Chỉ hỗ trợ bất phương trình có một dấu quan hệ.")
    symbols = tuple(sorted(set(before.symbols) | set(after.symbols), key=str))
    if len(symbols) != 1:
        raise _Unsupported("Bất phương trình phải có đúng một biến.")
    from sympy import Ge, Gt, Le, Lt
    from sympy.solvers.inequalities import solve_univariate_inequality

    constructors = {"<": Lt, "<=": Le, ">": Gt, ">=": Ge}
    if before.relations[0] not in constructors or after.relations[0] not in constructors:
        raise _Unsupported("Dấu quan hệ không được hỗ trợ.")
    variable = symbols[0]
    first = constructors[before.relations[0]](before.parts[0], before.parts[1])
    second = constructors[after.relations[0]](after.parts[0], after.parts[1])
    return bool(
        solve_univariate_inequality(first, variable, relational=False)
        == solve_univariate_inequality(second, variable, relational=False)
    )


def _is_simple_numeric_assignment(claim: _Claim) -> bool:
    from sympy import simplify

    if len(claim.relations) != 1 or claim.relations[0] != "=" or len(claim.symbols) != 1:
        return False
    variable = claim.symbols[0]
    return bool(
        (
            simplify(claim.parts[0] - variable) == 0
            and not claim.parts[1].free_symbols
        )
        or (
            simplify(claim.parts[1] - variable) == 0
            and not claim.parts[0].free_symbols
        )
    )


def _expression_operation_count(normalized_before: str) -> tuple[int, list[str]]:
    operations: list[str] = []
    if "**" in normalized_before:
        operations.append("expand_power")
    if re.search(r"(?:^|[+\-(])\d+\*", normalized_before):
        operations.append("distribute_coefficient")
    if re.search(r"\*\*\d+\)?[+\-]\d+$", normalized_before):
        operations.append("combine_constant_terms")
    if not operations:
        operations.append("simplify_expression")
    return len(operations), operations


def analyze_code_transition(*, before: str, after: str) -> dict[str, Any]:
    """Return one validated internal annotation for a canonical transition."""

    try:
        raw_before = str(before or "")
        raw_after = str(after or "")
        standalone_finding = _standalone_numeric_assertion(raw_after)
        if standalone_finding is not None:
            return standalone_finding.analysis
        raw_transition = raw_before + "\n" + raw_after
        if _DIVISION_BY_ZERO.search(raw_after):
            return _analysis(
                "verified_invalid",
                "equation_transformation",
                issue_type="division_by_zero",
                reason="Transition thực hiện phép chia hai vế cho 0.",
            )
        if _UNSAFE_DOMAIN.search(raw_transition):
            raise _Unsupported("Transition chứa căn, log hoặc hàm ngoài phạm vi an toàn.")
        if _VARIABLE_DIVISION.search(raw_transition):
            raise _Unsupported("Phép chia cho biểu thức chứa biến cần Gemma kiểm tra điều kiện.")
        explicit_chain = _analyze_explicit_claim_chain(raw_before, raw_after)
        if explicit_chain is not None:
            return explicit_chain
        before_claim = _extract_claim(before)
        after_claim = _extract_claim(
            after,
            allow_numeric_fragment=_is_simple_numeric_assignment(before_claim),
        )
        all_relations = before_claim.relations + after_claim.relations
        has_inequality = any(relation in {"<", "<=", ">", ">="} for relation in all_relations)
        has_equality = bool(before_claim.relations or after_claim.relations)

        if has_inequality:
            transition_type = "inequality_transformation"
            equivalent = _inequality_equivalent(before_claim, after_claim)
            if not equivalent:
                return _analysis(
                    "verified_invalid",
                    transition_type,
                    issue_type="inequality_direction_error",
                    reason="Hai bất phương trình không có cùng tập nghiệm.",
                )
            operation = _operation_count(before_claim, after_claim)
        elif has_equality:
            if (
                not before_claim.relations
                and after_claim.relations == ("=",)
            ):
                from sympy import simplify

                source = before_claim.parts[0]
                source_matches_left = simplify(source - after_claim.parts[0]) == 0
                source_matches_right = simplify(source - after_claim.parts[1]) == 0
                if not source_matches_left and not source_matches_right:
                    raise _Unsupported(
                        "Không có verifier chứng minh đẳng thức sau là phép rút gọn của biểu thức nguồn."
                    )
                if simplify(after_claim.parts[0] - after_claim.parts[1]) != 0:
                    return _analysis(
                        "verified_invalid",
                        "expression_transformation",
                        issue_type="invalid_equivalence",
                        reason="Đẳng thức rút gọn không tương đương biểu thức nguồn.",
                    )
                operation_count, operation_types = _expression_operation_count(
                    before_claim.normalized
                )
                return _analysis(
                    (
                        "compressed_but_equivalent"
                        if operation_count > 1
                        else "verified_valid"
                    ),
                    "expression_transformation",
                    operation_count=operation_count,
                    operation_types=operation_types,
                    reason=(
                        "Đẳng thức tương đương nhưng gộp nhiều thao tác khai triển/rút gọn."
                        if operation_count > 1
                        else "Code xác minh phép biến đổi biểu thức đơn giản."
                    ),
                )
            if _is_simple_numeric_assignment(before_claim) and not after_claim.symbols:
                transition_type = "numeric_substitution"
                variable = before_claim.symbols[0]
                assignment_value = _simple_assignment_value(before_claim, variable)
                assignment_text = f"{variable}={assignment_value}"
                if assignment_value is None or assignment_text not in _normalize_math(raw_after):
                    raise _Unsupported(
                        "Verifier thế số yêu cầu state sau lặp lại đúng assignment nguồn."
                    )
                if not _numeric_chain_valid(after_claim):
                    return _analysis(
                        "verified_invalid",
                        transition_type,
                        issue_type="calculation_error",
                        reason="Kết quả thay số hoặc chuỗi số học không đúng.",
                    )
                return _analysis(
                    "verified_valid",
                    transition_type,
                    operation_count=1,
                    operation_types=["substitute_numeric_value"],
                    reason="Code xác minh phép thay số đơn giản và kết quả số học.",
                )
            is_numeric_calculation = (
                before_claim.symbols
                and not before_claim.parts[-1].free_symbols
                and not after_claim.parts[-1].free_symbols
                and len(before_claim.symbols) == 1
                and str(before_claim.symbols[0]).isupper()
                and _is_simple_numeric_assignment(before_claim)
                and _is_simple_numeric_assignment(after_claim)
            )
            transition_type = (
                "equality_chain"
                if len(after_claim.relations) > 1
                else "numeric_calculation"
                if is_numeric_calculation
                else "equation_transformation"
            )
            if not is_numeric_calculation:
                _ensure_supported_polynomial_equations(before_claim, after_claim)
            if not _numeric_chain_valid(after_claim):
                return _analysis(
                    "verified_invalid",
                    transition_type,
                    issue_type="calculation_error",
                    reason="Chuỗi đẳng thức chứa quan hệ số học sai.",
                )
            equivalent = _equation_relation(before_claim, after_claim)
            if not equivalent:
                issue_type = (
                    "sign_error"
                    if _looks_like_sign_error(before_claim, after_claim)
                    else "calculation_error"
                    if transition_type == "numeric_calculation"
                    else "invalid_equivalence"
                )
                return _analysis(
                    "verified_invalid",
                    transition_type,
                    issue_type=issue_type,
                    reason="Transition không bảo toàn quan hệ của claim nguồn.",
                )
            operation = (
                (1, ["evaluate_numeric_expression"])
                if is_numeric_calculation
                else _operation_count(before_claim, after_claim)
            )
        else:
            if (
                tuple(str(symbol) for symbol in before_claim.symbols)
                != tuple(str(symbol) for symbol in after_claim.symbols)
                or len(before_claim.symbols) != 1
            ):
                raise _Unsupported(
                    "Không có verifier biểu thức cho transition đổi tập biến hoặc không có đúng một biến."
                )
            transition_type = (
                "numeric_calculation"
                if not before_claim.symbols and not after_claim.symbols
                else "expression_transformation"
            )
            from sympy import simplify

            equivalent = simplify(before_claim.parts[0] - after_claim.parts[0]) == 0
            if not equivalent:
                return _analysis(
                    "verified_invalid",
                    transition_type,
                    issue_type="calculation_error",
                    reason="Giá trị hoặc biểu thức sau không tương đương biểu thức trước.",
                )
            operation = (1, ["simplify_expression"])

        if operation is None:
            raise _Ambiguous("Transition tương đương nhưng chưa đếm chắc chắn được thao tác.")
        operation_count, operation_types = operation
        if operation_count > 1:
            return _analysis(
                "compressed_but_equivalent",
                transition_type,
                operation_count=operation_count,
                operation_types=operation_types,
                reason="Hai đầu transition tương đương nhưng cần nhiều thao tác đại số cơ bản.",
            )
        return _analysis(
            "verified_valid",
            transition_type,
            operation_count=operation_count,
            operation_types=operation_types,
            reason="Code xác minh transition hợp lệ trong phạm vi hỗ trợ.",
        )
    except _Unsupported as exc:
        return _analysis(
            "unsupported",
            "semantic_reasoning",
            reason=str(exc),
        )
    except _Ambiguous as exc:
        return _analysis(
            "ambiguous",
            "unsupported",
            reason=str(exc),
        )
    except (_ParseError, ValidationError) as exc:
        return _analysis(
            "parse_error",
            "unsupported",
            reason=str(exc),
        )
    except Exception as exc:  # fail open to Gemma, never crash the pipeline
        return _analysis(
            "parse_error",
            "unsupported",
            reason=f"Code analyzer không xử lý được transition: {exc}",
        )


def analyze_transition_stages(stage_payload: dict[str, Any]) -> dict[str, Any]:
    """Annotate all validated canonical stages without changing their anchors."""

    stages = stage_payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Ordered transitions phải có stages dạng list.")
    analyzed: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Mỗi transition phải là object canonical.")
        before = stage.get("_math_before", stage.get("bieu_thuc_truoc"))
        after = stage.get("_math_after", stage.get("bieu_thuc_sau"))
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("Transition thiếu before/after dạng text.")
        mandatory_finding = _standalone_numeric_assertion(after)
        analysis = (
            mandatory_finding.analysis
            if mandatory_finding is not None
            else analyze_code_transition(before=before, after=after)
        )
        analyzed_stage = {
            key: value
            for key, value in stage.items()
            if key not in {"_math_before", "_math_after"}
        }
        analyzed_stage["code_analysis"] = analysis
        if (
            mandatory_finding is not None
            and mandatory_finding.analysis.get("status") == "verified_invalid"
        ):
            analyzed_stage["mandatory_code_issue"] = {
                "evidence_text": mandatory_finding.evidence_text,
                "expected_result": mandatory_finding.expected_result,
                "actual_result": mandatory_finding.actual_result,
            }
        analyzed.append(analyzed_stage)
    return {"stages": analyzed}


__all__ = ["analyze_code_transition", "analyze_transition_stages"]
