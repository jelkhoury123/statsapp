"""Hover/focus tooltips for common regression abbreviations."""

from __future__ import annotations

import re

# Canonical display names (use these in UI copy). Subscripts via tip_visible().
# Tuple is (tooltip title, definition). Title may expand the short label.
STAT_TIPS: dict[str, tuple[str, str]] = {
    "SS Total": (
        "SS Total (total sum of squares)",
        "Total variation around the sample mean.",
    ),
    "SS Model": (
        "SS Model (model sum of squares)",
        "Variation explained by the regression.",
    ),
    "SSR": (
        "SSR — Residual Sum of Squares",
        "Sum of squared differences between observed and predicted values. "
        "lm() minimizes SSR.",
    ),
    "R²": (
        "Coefficient of Determination",
        "Proportion of total variation accounted for by the model "
        "(SS Model / SS Total). Compares regression predictions with "
        "the baseline that predicts every outcome by the sample mean Ȳ.",
    ),
    "Residual SE": (
        "Residual Standard Error",
        "Typical size of residuals around the fitted line "
        "(√(SSR / (n − 2)) after OLS).",
    ),
    "OLS": (
        "Ordinary Least Squares",
        "Method that finds the intercept and slope minimizing SSR "
        "(the residual sum of squares). Squaring prevents positive and "
        "negative errors from cancelling and weights large errors more heavily.",
    ),
}

# Optional third-line formulas shown in polished HTML tooltips only.
STAT_TIP_FORMULAS: dict[str, str] = {
    "SSR": "SSR = Σ(yᵢ − ŷᵢ)²",
}

# Legacy abbreviations still tipified if they appear; map to canonical labels.
_LEGACY_TIP_ALIASES: dict[str, str] = {
    "SST": "SS Total",
    "TSS": "SS Total",
    "SSE": "SSR",
    "RSS": "SSR",
    "SS Residual": "SSR",
    "SSM": "SS Model",
    "ESS": "SS Model",
    "RSE": "Residual SE",
}

# Longest codes first so "SS Residual" / "Residual SE" / "SS Total" win over fragments.
_TIP_CODES: tuple[str, ...] = tuple(
    sorted(
        (*STAT_TIPS.keys(), *_LEGACY_TIP_ALIASES.keys()),
        key=len,
        reverse=True,
    )
)

_TIP_SPAN_RE = re.compile(
    r'<span class="stat-tip"[^>]*>.*?<span class="stat-tip-bubble"[^>]*>.*?</span></span>',
    flags=re.DOTALL,
)


def resolve_tip_code(code: str) -> str:
    """Map a tip key (including legacy SST/SSE/SS Residual) to a STAT_TIPS entry."""
    return _LEGACY_TIP_ALIASES.get(code, code)


def tip_visible(code: str) -> str:
    """Visible label; SS terms use HTML subscripts when shown in tipify HTML."""
    if code == "RSE":
        return "RSE"
    canonical = resolve_tip_code(code)
    if canonical == "SSR":
        return "SSR"
    if canonical.startswith("SS "):
        return f"SS<sub>{canonical[3:]}</sub>"
    return canonical


def tip_title(code: str) -> str:
    """Single-line title suitable for native title=/help= attributes."""
    canonical = resolve_tip_code(code)
    full, detail = STAT_TIPS[canonical]
    formula = STAT_TIP_FORMULAS.get(canonical)
    if formula:
        return f"{full}. {detail} {formula}"
    return f"{full}. {detail}"


def tip_hover(code: str) -> str:
    """Compact label for Plotly hover lines (plain text, no HTML)."""
    canonical = resolve_tip_code(code)
    full, detail = STAT_TIPS[canonical]
    formula = STAT_TIP_FORMULAS.get(canonical)
    if full == canonical:
        text = f"{canonical} — {detail}"
    elif full.startswith(f"{canonical} —") or full.startswith(f"{canonical} ("):
        text = f"{full}. {detail}"
    else:
        text = f"{canonical} — {full}. {detail}"
    if formula:
        text = f"{text} {formula}"
    return text


def tip_html(code: str) -> str:
    """Focusable abbreviation with a compact custom tooltip bubble."""
    canonical = resolve_tip_code(code)
    full, detail = STAT_TIPS[canonical]
    formula = STAT_TIP_FORMULAS.get(canonical)
    formula_bit = f"<br>{formula}" if formula else ""
    return (
        f'<span class="stat-tip" tabindex="0">'
        f"{tip_visible(code)}"
        f'<span class="stat-tip-bubble" role="tooltip">'
        f"<strong>{full}</strong><br>{detail}{formula_bit}"
        f"</span></span>"
    )


def tipify(text: str) -> str:
    """Wrap known abbreviations in tip_html, without re-wrapping existing tips."""
    if not text:
        return text
    protected: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\0TIP{len(protected) - 1}\0"

    out = text
    for code in _TIP_CODES:
        # Shield tips already inserted so shorter codes (e.g. SSR) do not nest.
        out = _TIP_SPAN_RE.sub(_protect, out)
        if code not in out:
            continue
        out = out.replace(code, tip_html(code))
    out = _TIP_SPAN_RE.sub(_protect, out)
    for index, span in enumerate(protected):
        out = out.replace(f"\0TIP{index}\0", span)
    return out
