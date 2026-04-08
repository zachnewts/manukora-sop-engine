# Manukora S&OP Briefing Engine

A Python tool that generates a monthly S&OP (Sales & Operations Planning) briefing an executive can read in 5 minutes. It loads inventory data, pre-computes all analytics, feeds enriched data to Claude for inference and recommendations, and renders the output as a styled HTML report.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
python sop_engine.py --data data/mock_inventory.csv --output output/ --month "March 2026"
```

Output lands in `output/` as both `.md` and `.html` files.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to inventory CSV |
| `--output` | `output/` | Output directory |
| `--format` | `both` | `html`, `md`, or `both` |
| `--month` | `March 2026` | Report month for briefing header |
| `--model` | `claude-opus-4-20250514` | Claude model override |

## Architecture

```
CSV Data → [Pandas Metric Engine] → Enriched DataFrame → [Claude API] → Markdown → [HTML Renderer] → Styled Briefing
```

**The key design decision: separate math from reasoning.**

LLMs are unreliable at arithmetic but excellent at inference, synthesis, and narrative. This tool pre-computes every derived metric (cover months, trend percentages, revenue at risk, reorder urgency scores) in pandas, then feeds the enriched data to Claude. The LLM's job is to do what it's actually good at — interpret the numbers, make recommendations, and write a briefing a CEO can act on.

### What the data engine computes

| Metric | Formula | Purpose |
|--------|---------|---------|
| Combined demand (M1–M4) | Shopify + Amazon per month | Pooled channel demand |
| Current sell-through rate | M4 combined | Baseline for projections |
| Trend % | M1→M4 change (M2→M4 for new products) | Demand trajectory |
| Months of cover (raw) | Stock on Hand / current rate | Cover without incoming |
| Effective cover | (Stock + arriving orders) / current rate | Cover with incoming |
| Revenue at risk | Monthly revenue × cover shortfall | Dollar exposure |
| Reorder priority score | Revenue at risk × urgency multiplier | Composite ranking |

### What Claude does

- Interprets trends and their business implications
- Prioritizes reorder recommendations by revenue opportunity
- Flags conflicts (e.g., declining demand on a high-revenue SKU)
- Writes an executive-ready narrative with specific numbers and actions

## Prompt Design

The prompt architecture uses two layers:

### System Prompt
Sets the persona (senior S&OP analyst) and output standards:
- Every recommendation must include specific numbers, dollar impact, clear action, and priority rank
- Inference and judgment, not data description
- Direct language — no hedging or filler
- Includes a concrete example of expected output quality (the "good output" benchmark from the assignment)

### User Prompt
Provides everything the model needs in one structured payload:
1. **Reporting context** — time period, generation date
2. **Business rules** — product-specific exceptions the model must follow (Bioactive launch window, Propolis phase-out, MGO 1700+ special cover target, order arrival semantics)
3. **Enriched data table** — all 12 SKUs with ~25 pre-computed columns
4. **Output format** — exact section structure with specific instructions per section

This structure ensures the model has verified numbers to reference and clear constraints to follow, while focusing its generation on the high-value work: synthesis, prioritization, and actionable recommendations.

## Business Rules Encoded

- **Bioactive Blends** (Immunity, Energy, Recovery): Launched mid-January 2026. Trend computed M2→M4 only. Flagged as new products.
- **Propolis Tincture 30ml**: Phase-out Q2 2026. Deprioritized for reorder unless cover < 30 days.
- **MGO 1700+ 100g**: 3-month target cover (not 2) due to premium price and longer lead times.
- **Order_Arrival_Months = 0**: No order placed (not immediate arrival). Only orders with arrival > 0 count toward effective cover.
- **Inventory pool**: Single pooled inventory serves both Shopify and Amazon channels.

## Extending This

In a production environment, this architecture connects naturally to real systems:

- **Data source**: Replace CSV loading with Shopify API + Cin7/OMS inventory feeds
- **Scheduling**: Cron job or Airflow DAG runs monthly, outputs to Slack/email
- **Historical context**: Feed previous month's briefing as additional context for trend narrative
- **Multi-format**: Add PDF export via `weasyprint`, Slack block formatting
- **Alerting**: Threshold-based triggers (cover drops below 1 month → immediate Slack alert, don't wait for monthly briefing)

## File Structure

```
prep/assignment/
├── sop_engine.py              # Single-file tool: load → compute → prompt → render
├── data/
│   └── mock_inventory.csv     # Mock data (12 SKUs, 14 columns)
├── output/                    # Generated briefings
├── requirements.txt           # anthropic, pandas, markdown
└── README.md                  # This file
```
