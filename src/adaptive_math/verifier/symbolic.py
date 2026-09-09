"""Symbolic expression comparison.

This module is designed to run ONLY inside the isolated verifier worker
process (verifier.worker): it parses untrusted LaTeX with math-verify,
compares with SymPy and cross-checks numerically at deterministic points.
It never imports model, agent or network code and never executes input.
"""

import hashlib
import math
import random

import sympy
from math_verify.parser import parse_latex_cached

from adaptive_math.verifier.normalizer import normalize_surface

MAX_EXPR_CHARS = 8192
MAX_NODE_COUNT = 4096
MAX_FREE_SYMBOLS = 32
REQUIRED_POINTS = 5
MAX_POINT_ATTEMPTS = 200
POINT_RANGE = 5.0
CROSS_ATOL = 1e-6
CROSS_RTOL = 1e-6

_APPLIED_UNDEF = sympy.core.function.AppliedUndef


def compare_expressions(prediction: str, reference: str, task_id: str) -> dict[str, object]:
    """Compare two LaTeX expressions and return a VerifierResult-shaped dict.

    Never raises. Order of evidence: canonical equality, domain-aware SymPy
    simplify difference, then a numeric cross-check at deterministic
    non-singular points seeded from task_id. The cross-check only returns
    correct when both expressions are defined at all points with error below
    tolerance; if expressions cannot be evaluated at all, the prediction is
    unverifiable (invalid) rather than guessed incorrect.
    """
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference)
    if len(pred_norm) > MAX_EXPR_CHARS:
        return _verdict("invalid_prediction", reason="expression exceeds length limit")
    if len(ref_norm) > MAX_EXPR_CHARS:
        return _verdict("invalid_reference", reason="expression exceeds length limit")
    pred_sym = _parse_to_sympy(pred_norm)
    if pred_sym is None:
        return _verdict("invalid_prediction", reason="malformed expression")
    ref_sym = _parse_to_sympy(ref_norm)
    if ref_sym is None:
        return _verdict("invalid_reference", reason="malformed expression")
    if _node_count(pred_sym) > MAX_NODE_COUNT:
        return _verdict("invalid_prediction", reason="expression has too many nodes")
    if _node_count(ref_sym) > MAX_NODE_COUNT:
        return _verdict("invalid_reference", reason="expression has too many nodes")
    if len(pred_sym.free_symbols) > MAX_FREE_SYMBOLS:
        return _verdict("invalid_prediction", reason="too many free symbols")
    if len(ref_sym.free_symbols) > MAX_FREE_SYMBOLS:
        return _verdict("invalid_reference", reason="too many free symbols")
    if pred_sym.atoms(_APPLIED_UNDEF) or ref_sym.atoms(_APPLIED_UNDEF):
        # unknown functions like f(x) cannot be verified reliably
        if pred_sym.atoms(_APPLIED_UNDEF):
            return _verdict("invalid_prediction", reason="unknown function application")
        return _verdict("invalid_reference", reason="unknown function application")

    pred_canon = str(pred_sym)
    ref_canon = str(ref_sym)
    if pred_canon == ref_canon:
        return _verdict(
            "correct",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "canonical"},
        )

    try:
        diff = sympy.simplify(pred_sym - ref_sym)
    except Exception:
        diff = pred_sym - ref_sym
    if diff == 0:
        if _domains_agree(pred_sym, ref_sym):
            return _verdict(
                "correct",
                normalized_prediction=pred_canon,
                normalized_reference=ref_canon,
                details={"method": "simplify"},
            )
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "simplify", "reason": "domains differ"},
        )

    if pred_sym.free_symbols != ref_sym.free_symbols:
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "structure", "reason": "free symbol mismatch"},
        )
    return _numeric_cross_check(pred_sym, ref_sym, pred_canon, ref_canon, task_id)


def _parse_to_sympy(text: str) -> sympy.Expr | None:
    """Parse via math-verify's LaTeX-to-sympy layer (latex2sympy2).

    math-verify 0.9's high-level parse() extracts numeric answers from
    solution text and cannot handle symbolic expressions; the parsing layer
    underneath it (parse_latex_cached) is the right entry point for
    already-normalized expressions.
    """
    try:
        parsed = parse_latex_cached(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, sympy.Expr) else None


def _node_count(expr: sympy.Expr) -> int:
    count = 0
    for _ in sympy.preorder_traversal(expr):
        count += 1
        if count > MAX_NODE_COUNT:
            break
    return count


def _domains_agree(a: sympy.Expr, b: sympy.Expr) -> bool:
    """Compare singularities over every shared free symbol. A simplify
    difference of zero only proves equality where both sides are defined."""
    for symbol in sorted(a.free_symbols | b.free_symbols, key=str):
        try:
            if sympy.singularities(a, symbol) != sympy.singularities(b, symbol):
                return False
        except (NotImplementedError, TypeError, AttributeError):
            continue  # undecidable here; rely on the simplify evidence
    return True


def _numeric_cross_check(
    pred_sym: sympy.Expr,
    ref_sym: sympy.Expr,
    pred_canon: str,
    ref_canon: str,
    task_id: str,
) -> dict[str, object]:
    symbols = sorted(pred_sym.free_symbols, key=str)
    seed = hashlib.sha256(f"{task_id}\x00{pred_canon}\x00{ref_canon}".encode()).digest()
    rng = random.Random(seed)
    points: list[tuple[dict[sympy.Symbol, float], float]] = []
    attempts = 0
    while len(points) < REQUIRED_POINTS and attempts < MAX_POINT_ATTEMPTS:
        attempts += 1
        subs = {symbol: rng.uniform(-POINT_RANGE, POINT_RANGE) for symbol in symbols}
        try:
            p_val = _evaluate(pred_sym, subs)
            r_val = _evaluate(ref_sym, subs)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue  # singular or unrepresentable point; try the next one
        if p_val is None or r_val is None:
            continue
        error = abs(p_val - r_val)
        if not math.isfinite(error):
            continue
        points.append((subs, error))
    recorded_points: dict[str, object] = {
        str(symbol): [round(subs[symbol], 6) for subs, _ in points] for symbol in symbols
    }
    if len(points) < REQUIRED_POINTS:
        return _verdict(
            "invalid_prediction",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={
                "method": "numeric",
                "reason": "expressions could not be evaluated at sample points",
                "points": recorded_points,
            },
        )
    max_error = max(error for _, error in points)
    max_ref = max(abs(_evaluate(ref_sym, subs) or 0.0) for subs, _ in points)
    tolerance = CROSS_ATOL + CROSS_RTOL * max_ref
    if max_error <= tolerance:
        return _verdict(
            "correct",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={
                "method": "numeric",
                "points": recorded_points,
                "max_error": max_error,
            },
        )
    return _verdict(
        "incorrect",
        normalized_prediction=pred_canon,
        normalized_reference=ref_canon,
        details={
            "method": "numeric",
            "points": recorded_points,
            "max_error": max_error,
        },
    )


def _evaluate(expr: sympy.Expr, subs: dict[sympy.Symbol, float]) -> complex | None:
    value = sympy.N(expr.subs(subs), 15)
    try:
        re_part, im_part = value.as_real_imag()
        result = complex(float(re_part), float(im_part))
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        return None
    return result


def _verdict(
    status: str,
    *,
    reason: str | None = None,
    normalized_prediction: str | None = None,
    normalized_reference: str | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(details or {})
    if reason:
        merged["reason"] = reason
    return {
        "status": status,
        "reward": 1.0 if status == "correct" else 0.0,
        "normalized_prediction": normalized_prediction,
        "normalized_reference": normalized_reference,
        "details": merged,
    }
