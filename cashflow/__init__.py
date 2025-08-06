from __future__ import annotations
import ast
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None

__all__ = [
    "analyze",
    "evaluate_rules",
    "render_markdown",
    "render_html",
]


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["date", "description", "amount", "balance"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}; found {list(df.columns)}")
    df = df.rename(columns={cols[c]: c for c in required})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).sort_values("date")
    return df.reset_index(drop=True)


def _daily_views(df: pd.DataFrame) -> Dict[str, pd.Series]:
    daily_net = df.groupby(df["date"].dt.date)["amount"].sum().sort_index()
    daily_eod = df.sort_values(["date"]).groupby(df["date"].dt.date)["balance"].last()
    return {"daily_net": daily_net, "daily_eod": daily_eod}


def _monthly_gross_revenue(df: pd.DataFrame) -> Dict[str, float]:
    credits = df[df["amount"] > 0].copy()
    if credits.empty:
        return {}
    grp = credits.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    return {str(p): round(float(v), 2) for p, v in grp.items()}


def _nsf_count(df: pd.DataFrame) -> int:
    desc = df["description"].str.lower()
    return int(desc.str.contains("nsf|non[- ]?sufficient").sum())


def _nsf_count_last_90d(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    end = df["date"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    start = end - pd.Timedelta(days=90)
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    return _nsf_count(window)


def _inflow_outflow_ratio(df: pd.DataFrame) -> float:
    inflow = df.loc[df["amount"] > 0, "amount"].sum()
    outflow = df.loc[df["amount"] < 0, "amount"].sum()
    denom = abs(outflow) if outflow != 0 else float("inf")
    ratio = float(inflow) / denom if denom != float("inf") else float("inf")
    return round(ratio, 4)


def _volatility_std(daily_net: pd.Series) -> float:
    if daily_net.empty:
        return 0.0
    return round(float(daily_net.std(ddof=0)), 2)


def _rolling_90_day_trend(daily_net: pd.Series) -> Dict[str, Any]:
    if daily_net.empty:
        return {"last_90d_net_inflow": 0.0, "prev_90d_net_inflow": 0.0, "pct_change": 0.0}
    idx = pd.date_range(start=daily_net.index.min(), end=daily_net.index.max(), freq="D")
    series = daily_net.reindex(idx.date, fill_value=0.0)
    roll = series.rolling(window=90, min_periods=1).sum()
    if len(roll) < 91:
        prev = 0.0
    else:
        prev = float(roll.iloc[-91])
    last = float(roll.iloc[-1])
    pct = (last - prev) / (abs(prev) if prev != 0 else 1.0)
    return {
        "last_90d_net_inflow": round(last, 2),
        "prev_90d_net_inflow": round(prev, 2),
        "pct_change": round(pct, 4),
    }


def _seasonality_hint(monthly_rev: Dict[str, float]) -> str:
    if not monthly_rev:
        return "no obvious seasonality"
    vals = pd.Series(monthly_rev)
    if len(vals) < 3:
        return "insufficient data for seasonality"
    cv = vals.std(ddof=0) / (vals.mean() if vals.mean() else 1.0)
    if cv > 0.5:
        median = vals.median()
        spikes = [k for k, v in vals.items() if v >= 1.2 * median]
        return f"high variability; spikes in: {', '.join(spikes)}"
    return "low variability"


def analyze(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = _coerce_columns(df)
    views = _daily_views(df)

    monthly_rev = _monthly_gross_revenue(df)
    avg_daily_balance = float(views["daily_eod"].mean()) if not views["daily_eod"].empty else 0.0
    nsf_total = _nsf_count(df)
    nsf_90d = _nsf_count_last_90d(df)
    ior = _inflow_outflow_ratio(df)
    vol_std = _volatility_std(views["daily_net"])
    trend = _rolling_90_day_trend(views["daily_net"])
    seasonality = _seasonality_hint(monthly_rev)

    start_date = df["date"].min().date().isoformat() if not df.empty else None
    end_date = df["date"].max().date().isoformat() if not df.empty else None

    metrics = {
        "as_of": end_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "period": {"start": start_date, "end": end_date},
        "days_covered": int((df["date"].max().date() - df["date"].min().date()).days + 1) if not df.empty else 0,
        "monthly_gross_revenue": monthly_rev,
        "avg_daily_balance": round(avg_daily_balance, 2),
        "nsf_count": int(nsf_total),
        "nsf_count_90d": int(nsf_90d),
        "inflow_outflow_ratio": ior,
        "volatility_std": vol_std,
        "rolling_90_day_trend": trend,
        "seasonality_hint": seasonality,
    }
    return metrics


ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name,
    ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.USub, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
)


def _safe_eval(expr: str, context: Dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in context:
            raise ValueError(f"Unknown name in expression: {node.id}")
    code = compile(tree, filename="<rules>", mode="eval")
    return bool(eval(code, {"__builtins__": {}}, context))


@dataclass
class RuleResult:
    name: str
    condition: str
    passed: bool
    message: Optional[str] = None


def evaluate_rules(metrics: Dict[str, Any], rules_yaml_path: Optional[str]) -> List[RuleResult]:
    if not rules_yaml_path or yaml is None:
        return []
    with open(rules_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = data.get("rules", [])

    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
    flat.setdefault("monthly_gross_revenue_total", float(sum((metrics.get("monthly_gross_revenue") or {}).values())))

    results: List[RuleResult] = []
    for r in rules:
        name = r.get("name", "unnamed")
        cond = r.get("condition", "")
        msg = r.get("message")
        try:
            hit = _safe_eval(cond, flat)
        except Exception as e:
            hit = False
            msg = f"Invalid rule '{name}': {e}"
        results.append(RuleResult(name=name, condition=cond, passed=bool(hit), message=msg))
    return results


def render_markdown(metrics: Dict[str, Any], flags: List[RuleResult] | None = None) -> str:
    flags = flags or []
    lines = ["# Cash-Flow KPIs", ""]
    period = metrics.get("period", {})
    lines.append(f"**Period:** {period.get('start')} → {period.get('end')}  ")
    lines.append(f"**Days covered:** {metrics.get('days_covered')}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Avg daily balance: **${metrics.get('avg_daily_balance'):,}**")
    lines.append(f"- NSFs (total / 90d): **{metrics.get('nsf_count')} / {metrics.get('nsf_count_90d')}**")
    lines.append(f"- Inflow/Outflow ratio: **{metrics.get('inflow_outflow_ratio')}**")
    lines.append(f"- Volatility (std of daily net): **${metrics.get('volatility_std'):,}**")
    trend = metrics.get("rolling_90_day_trend", {})
    lines.append(
        f"- 90d net inflow: **${trend.get('last_90d_net_inflow', 0):,}** (prev **${trend.get('prev_90d_net_inflow', 0):,}**, Δ {trend.get('pct_change', 0)*100:.1f}%)"
    )
    lines.append(f"- Seasonality: _{metrics.get('seasonality_hint')}_")
    lines.append("")

    lines.append("## Monthly Gross Revenue")
    mrev = metrics.get("monthly_gross_revenue", {})
    if mrev:
        lines.append("| Month | Gross Inflow |")
        lines.append("|---|---:|")
        for k, v in sorted(mrev.items()):
            lines.append(f"| {k} | ${v:,.2f} |")
    else:
        lines.append("(no positive inflows detected)")

    if flags:
        lines.append("")
        lines.append("## Fundability Flags")
        for fr in flags:
            if fr.passed:
                lines.append(f"- ❗ **{fr.name}** — {fr.message or fr.condition}")

    lines.append("")
    lines.append(f"_Generated at {metrics.get('generated_at')}_")
    return "
".join(lines)


def render_html(markdown_text: str) -> str:
    import html
    body = "<br/>".join(html.escape(line) for line in markdown_text.split("
"))
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cash-Flow KPIs</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; line-height: 1.45; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
  th {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
</style>
<body>
{body}
</body>
</html>
"""
