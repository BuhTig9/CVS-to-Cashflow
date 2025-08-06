# csv-to-cashflow

Small Python CLI + library that reads a bank-statement CSV and emits MCA‑relevant cash‑flow KPIs as JSON. Includes a minimal YAML rules engine for fundability flags, plus optional Markdown and HTML summaries.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # or use uv/conda
pip install pandas pyyaml pytest
```

> This repo is a simple package folder. You can run it in-place without packaging.

## CSV format

Columns required (case-insensitive):

- `date` — parsable date/datetime
- `description` — free text
- `amount` — positive = inflow, negative = outflow
- `balance` — running balance after the transaction

## CLI

```bash
python -m cashflow.cli analyze cashflow/samples/healthy.csv \
  --rules cashflow/rules.yaml \
  --out metrics.json \
  --pretty \  --html report.html
```

- `--out` writes JSON (default stdout)
- `--pretty` also prints a Markdown summary
- `--html` writes a simple HTML report
- `--rules` evaluates YAML rules to produce **flags**

## Library usage

```python
from cashflow import analyze, evaluate_rules, render_markdown, render_html

metrics = analyze("cashflow/samples/healthy.csv")
flags = evaluate_rules(metrics, "cashflow/rules.yaml")
md = render_markdown(metrics, flags)
html = render_html(md)
```

## Tests

```bash
pytest -q
```

## Notes

- **NSF detection** is heuristic: any transaction whose description contains `NSF` is counted.
- `monthly_gross_revenue` is the sum of **positive** amounts per calendar month.
- `volatility_std` is the standard deviation of **daily net** cash flow.
- `rolling_90_day_trend` compares the last 90‑day net inflow vs one window prior.
- The rules engine supports simple boolean/arithmetic expressions over scalar KPIs.
