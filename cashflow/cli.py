from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import analyze, evaluate_rules, render_markdown, render_html


def _write_text(path: str | None, text: str):
    if not path or path == "-":
        print(text)
        return
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cashflow", description="CSV → cash-flow KPIs")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze a bank-statement CSV")
    a.add_argument("csv", help="Path to CSV (date, description, amount, balance)")
    a.add_argument("--out", default="-", help="Write JSON to file (default stdout)")
    a.add_argument("--indent", type=int, default=2, help="JSON indent (default 2)")
    a.add_argument("--rules", default=None, help="Path to rules.yaml (optional)")
    a.add_argument("--pretty", action="store_true", help="Also print Markdown summary")
    a.add_argument("--html", default=None, help="Additionally write HTML report to this path")

    args = p.parse_args(argv)

    if args.cmd == "analyze":
        metrics = analyze(args.csv)
        flags = evaluate_rules(metrics, args.rules)
        payload = {"metrics": metrics, "flags": [fr.__dict__ for fr in flags]}
        _write_text(args.out, json.dumps(payload, indent=args.indent))

        if args.pretty:
            md = render_markdown(metrics, flags)
            print("\n\n" + md)
        if args.html:
            md = render_markdown(metrics, flags)
            html = render_html(md)
            _write_text(args.html, html)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
