#!/usr/bin/env python3
"""
Manukora S&OP Briefing Engine
─────────────────────────────
Generates a monthly S&OP briefing an exec can read in 5 minutes.
Architecture: pre-compute analytics in pandas → feed enriched data to Claude → render styled HTML.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import anthropic
import markdown
import pandas as pd

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "SKU", "Shopify_M1", "Shopify_M2", "Shopify_M3", "Shopify_M4",
    "Amazon_M1", "Amazon_M2", "Amazon_M3", "Amazon_M4",
    "Stock_On_Hand", "Units_On_Order", "Order_Arrival_Months",
    "Target_Months_Cover", "Retail_Price_USD",
]

BIOACTIVE_SKUS = [
    "Bioactive Blend Immunity 250g",
    "Bioactive Blend Energy 250g",
    "Bioactive Blend Recovery 250g",
]

PHASEOUT_SKUS = ["Propolis Tincture 30ml"]

# ─────────────────────────────────────────────
# Task 1 — Data Loading
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load inventory CSV and validate expected columns."""
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "SKU"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────
# Task 2 — Metric Computation
# ─────────────────────────────────────────────

def classify_trend(pct_change: float) -> str:
    """Classify demand trend based on % change with ±5% threshold."""
    if pct_change > 5:
        return "accelerating"
    elif pct_change < -5:
        return "declining"
    return "stable"


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived S&OP metrics per SKU."""
    out = df.copy()

    # Combined channel demand per month
    for m in range(1, 5):
        out[f"combined_m{m}"] = out[f"Shopify_M{m}"] + out[f"Amazon_M{m}"]

    # Current sell-through rate = M4 combined
    out["current_rate"] = out["combined_m4"]

    # Month-over-month growth rates
    out["mom_m2_m1"] = (out["combined_m2"] - out["combined_m1"]) / out["combined_m1"] * 100
    out["mom_m3_m2"] = (out["combined_m3"] - out["combined_m2"]) / out["combined_m2"] * 100
    out["mom_m4_m3"] = (out["combined_m4"] - out["combined_m3"]) / out["combined_m3"] * 100

    # Trend % — M1→M4 for most SKUs, M2→M4 for Bioactive Blends (launched mid-Jan)
    trend_pcts = []
    for _, row in out.iterrows():
        if row["SKU"] in BIOACTIVE_SKUS:
            base = row["combined_m2"]
            current = row["combined_m4"]
        else:
            base = row["combined_m1"]
            current = row["combined_m4"]
        trend_pcts.append((current - base) / base * 100)
    out["trend_pct"] = trend_pcts
    out["trend_direction"] = out["trend_pct"].apply(classify_trend)

    # Arriving stock: only count if Order_Arrival_Months > 0 (0 = no order placed)
    out["arriving_stock"] = out.apply(
        lambda r: r["Units_On_Order"] if r["Order_Arrival_Months"] > 0 else 0, axis=1
    )

    # Stock cover calculations
    out["months_cover_raw"] = out["Stock_On_Hand"] / out["current_rate"]
    out["effective_cover"] = (out["Stock_On_Hand"] + out["arriving_stock"]) / out["current_rate"]
    out["cover_vs_target"] = out["effective_cover"] - out["Target_Months_Cover"]

    # Revenue metrics
    out["monthly_revenue"] = out["current_rate"] * out["Retail_Price_USD"]
    out["revenue_at_risk"] = out["monthly_revenue"] * out.apply(
        lambda r: max(0, r["Target_Months_Cover"] - r["effective_cover"]), axis=1
    )

    # Reorder priority score — higher = more urgent
    # Urgency multiplier: inverse of cover ratio, floored at 0
    out["urgency_multiplier"] = out.apply(
        lambda r: max(0, 1 - r["effective_cover"] / r["Target_Months_Cover"])
        if r["Target_Months_Cover"] > 0 else 0,
        axis=1,
    )
    out["reorder_priority_score"] = out["revenue_at_risk"] * out["urgency_multiplier"]

    # Deprioritize phase-out SKUs unless cover < 1 month (~30 days)
    for idx, row in out.iterrows():
        if row["SKU"] in PHASEOUT_SKUS and row["months_cover_raw"] >= 1.0:
            out.at[idx, "reorder_priority_score"] = 0.0

    # Flags
    out["is_bioactive"] = out["SKU"].isin(BIOACTIVE_SKUS)
    out["is_phaseout"] = out["SKU"].isin(PHASEOUT_SKUS)

    return out


# ─────────────────────────────────────────────
# Task 3 — Prompt Engineering
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior S&OP analyst at a premium DTC consumer brand (Manukora — premium Manuka honey). You write monthly briefings for the CEO.

Your output standards:
- Every recommendation MUST include: specific numbers, business impact in dollars, a clear action, and a priority rank.
- You do INFERENCE and JUDGMENT — not data description. Don't say "sales increased 23%." Say what that means for the business and what to do about it.
- When demand trends conflict with revenue materiality (e.g., a declining SKU still generating significant revenue), flag the conflict explicitly and state your recommendation.
- Write for a CEO who has 5 minutes. Lead with what matters most.
- Use the pre-computed metrics provided — they are verified. Do not re-derive or round them differently.
- Be direct. No hedging, no "it's worth noting," no filler. State the situation, the implication, and the action.

Example of the output quality expected for reorder recommendations:
"MGO 263+ 500g has ~2.5 months of cover at current combined sell-through, no stock on order, and demand has grown 23% over 4 months. At $54.99 retail and ~684 units/month combined, this is ~$37K/month in revenue at risk. Cover will drop below target before a new order could arrive — recommend ordering immediately. Priority 1."

Match this level of specificity and decisiveness throughout."""


def build_user_prompt(enriched_df: pd.DataFrame, report_month: str) -> str:
    """Build the user prompt with enriched data and output format instructions."""
    generation_date = datetime.now().strftime("%B %d, %Y")

    # Format the enriched data as a readable table for the LLM
    display_cols = [
        "SKU", "combined_m1", "combined_m2", "combined_m3", "combined_m4",
        "current_rate", "trend_pct", "trend_direction",
        "mom_m2_m1", "mom_m3_m2", "mom_m4_m3",
        "Stock_On_Hand", "arriving_stock", "Units_On_Order", "Order_Arrival_Months",
        "months_cover_raw", "effective_cover", "Target_Months_Cover", "cover_vs_target",
        "Retail_Price_USD", "monthly_revenue", "revenue_at_risk",
        "reorder_priority_score", "is_bioactive", "is_phaseout",
    ]
    data_table = enriched_df[display_cols].to_string(index=False)

    return f"""Generate a monthly S&OP briefing for {report_month}.

## Reporting Context
- Period: M1 = December 2025, M2 = January 2026, M3 = February 2026, M4 = March 2026 (most recent)
- Report generated: {generation_date}
- Data source: Combined Shopify + Amazon sales, pooled inventory

## Business Rules (critical — follow these exactly)
1. **Bioactive Blends** (Immunity, Energy, Recovery) launched mid-January 2026. Their M1 data is pre-launch and meaningless. Trend is computed M2→M4 only. Flag them as new products.
2. **Propolis Tincture 30ml** is being phased out Q2 2026. Flag stockout risk but deprioritize for reorder unless cover is under 30 days.
3. **MGO 1700+ 100g** has a 3-month target cover (not 2) due to premium price and longer lead times.
4. **Inventory pool:** All Shopify + Amazon orders draw from ONE pooled inventory. Stock_On_Hand is global.
5. **Order_Arrival_Months = 0** means NO ORDER PLACED, not immediate arrival. Only stock with Order_Arrival_Months > 0 is actually incoming.
6. **Sell-through baseline:** Use M4 (March 2026) as the current rate. Note clear trends if they change the reorder calculus.

## Enriched Data (all metrics pre-computed and verified)

{data_table}

Column definitions:
- combined_m1..m4: Shopify + Amazon units sold per month
- current_rate: M4 combined (current monthly sell-through)
- trend_pct: % change M1→M4 (M2→M4 for Bioactive Blends)
- mom_*: month-over-month growth rates (%)
- months_cover_raw: Stock_On_Hand / current_rate (without incoming)
- effective_cover: (Stock_On_Hand + arriving_stock) / current_rate
- cover_vs_target: effective_cover - Target_Months_Cover (negative = below target)
- revenue_at_risk: monthly_revenue × shortfall months (dollars exposed if stock runs out)
- reorder_priority_score: revenue_at_risk × urgency_multiplier (composite ranking)

## Output Format

Structure the briefing with these sections in this order:

### Executive Summary
3-4 bullets maximum. Lead with the single most important action item. Include headline wins, headline risks, and the top reorder action. A CEO should be able to read just this section and know what to do.

### Sales Performance
What's hot, what's not. Channel mix observations (Shopify vs Amazon shifts). Trend standouts — accelerating SKUs, decelerating SKUs, and any surprises. For Bioactive Blends, assess the launch trajectory (M2→M4 only).

### Stock Cover Risk Assessment
Every SKU with effective cover below target, sorted by severity (largest negative cover_vs_target first). For each: current cover, target, gap, incoming stock status, and what happens if no action is taken. Include timeline context — how many weeks/months until stockout at current rate.

### Reorder Recommendations
Minimum 3 SKUs, priority-ranked by revenue opportunity (not just cover risk). For each recommendation, include:
- Current cover and trend
- Revenue at risk (dollars)
- Whether stock is on order
- Specific action (order now, expedite, etc.)
- Priority rank with reasoning

### Trend Conflicts & Watch List
- SKUs where declining demand conflicts with still-material revenue
- New products (Bioactive Blends) — early trajectory assessment and what to watch for next month
- Propolis phase-out status and any action needed
- Any other signals worth flagging for next month's review"""


def generate_briefing(enriched_df: pd.DataFrame, report_month: str, model: str = "claude-opus-4-20250514") -> str:
    """Call Claude API with enriched data, return markdown briefing."""
    client = anthropic.Anthropic()
    user_prompt = build_user_prompt(enriched_df, report_month)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text


# ─────────────────────────────────────────────
# Task 4 — HTML Output Engine
# ─────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Manukora S&amp;OP Briefing — {report_month}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  :root {{
    --navy: #0f1a2e;
    --navy-mid: #1e2d45;
    --navy-light: #2d3a4a;
    --amber: #d4a843;
    --amber-bright: #e8b84a;
    --amber-light: #f5e6c0;
    --amber-dim: rgba(212, 168, 67, 0.10);
    --red: #dc3545;
    --red-bg: rgba(220, 53, 69, 0.07);
    --red-border: rgba(220, 53, 69, 0.20);
    --green: #198754;
    --green-bg: rgba(25, 135, 84, 0.07);
    --green-border: rgba(25, 135, 84, 0.20);
    --orange: #e67e22;
    --orange-bg: rgba(230, 126, 34, 0.07);
    --orange-border: rgba(230, 126, 34, 0.20);
    --blue: #3b82f6;
    --blue-bg: rgba(59, 130, 246, 0.07);
    --blue-border: rgba(59, 130, 246, 0.20);
    --purple: #7c3aed;
    --purple-bg: rgba(124, 58, 237, 0.07);
    --purple-border: rgba(124, 58, 237, 0.20);
    --gray-50: #f8f9fb;
    --gray-100: #f1f3f5;
    --gray-200: #e2e5e9;
    --gray-300: #cdd1d7;
    --gray-400: #9ca3af;
    --gray-500: #7b8290;
    --gray-600: #5f6673;
    --gray-900: #111827;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: "SF Mono", "Fira Code", "Fira Mono", "Cascadia Code", Menlo, monospace;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --radius: 10px;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.7;
    color: var(--navy);
    background: linear-gradient(160deg, #f0ebe3 0%, var(--gray-50) 40%, #eef1f5 100%);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }}

  .container {{
    max-width: 880px;
    margin: 0 auto;
    padding: 48px 24px 80px;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
    border-radius: var(--radius);
    padding: 36px 40px 32px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
  }}

  header::before {{
    content: "";
    position: absolute;
    top: -40px;
    right: -40px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(212,168,67,0.15) 0%, transparent 70%);
    pointer-events: none;
  }}

  header .brand {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 8px;
  }}

  header h1 {{
    font-size: 26px;
    font-weight: 800;
    color: #fff;
    line-height: 1.25;
    margin-bottom: 12px;
  }}

  header .meta {{
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }}

  header .meta span {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}

  /* ── Section Cards ── */
  .section {{
    background: #fff;
    border-radius: var(--radius);
    box-shadow: var(--shadow-md);
    padding: 0;
    margin-bottom: 16px;
    overflow: hidden;
    border: 1px solid rgba(0,0,0,0.04);
  }}

  .section-header {{
    padding: 20px 32px 16px;
    border-bottom: 1px solid var(--gray-100);
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .section-icon {{
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }}

  .section-body {{
    padding: 20px 32px 28px;
  }}

  /* Section color variants */
  .section--summary .section-header {{ background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%); border-bottom: none; }}
  .section--summary .section-header h2 {{ color: #fff; border: none; padding: 0; margin: 0; }}
  .section--summary .section-icon {{ background: rgba(212,168,67,0.2); }}
  .section--summary .section-body {{ background: var(--navy); color: rgba(255,255,255,0.88); padding: 0 32px 28px; }}
  .section--summary li {{ border-color: rgba(255,255,255,0.1); }}
  .section--summary strong {{ color: #fff; }}

  .section--sales .section-icon {{ background: var(--green-bg); border: 1px solid var(--green-border); }}
  .section--risk .section-icon {{ background: var(--red-bg); border: 1px solid var(--red-border); }}
  .section--reorder .section-icon {{ background: var(--orange-bg); border: 1px solid var(--orange-border); }}
  .section--watch .section-icon {{ background: var(--purple-bg); border: 1px solid var(--purple-border); }}

  h2 {{
    font-size: 16px;
    font-weight: 700;
    color: var(--navy);
    margin: 0;
    padding: 0;
    border: none;
    letter-spacing: -0.01em;
  }}

  h3 {{
    font-size: 14px;
    font-weight: 700;
    color: var(--navy);
    margin-top: 24px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--gray-100);
  }}

  .section-body > p:first-child {{ margin-top: 0; }}

  p {{ margin-bottom: 14px; }}

  strong {{ font-weight: 650; }}

  /* ── Lists ── */
  ul, ol {{
    margin: 8px 0 16px 0;
    padding: 0;
    list-style: none;
  }}

  li {{
    padding: 10px 16px;
    margin-bottom: 0;
    border-bottom: 1px solid var(--gray-100);
    position: relative;
    padding-left: 28px;
  }}

  li:last-child {{
    border-bottom: none;
  }}

  li::before {{
    content: "";
    position: absolute;
    left: 10px;
    top: 18px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--amber);
  }}

  ol {{
    counter-reset: li-counter;
  }}

  ol li {{
    padding-left: 36px;
  }}

  ol li::before {{
    content: counter(li-counter);
    counter-increment: li-counter;
    background: var(--amber);
    color: var(--navy);
    font-weight: 700;
    font-size: 11px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: absolute;
    left: 8px;
    top: 12px;
  }}

  li > ul, li > ol {{
    margin-top: 8px;
    margin-bottom: 4px;
    margin-left: 4px;
  }}

  li > ul li, li > ol li {{
    padding: 4px 8px 4px 24px;
    border-bottom: none;
    font-size: 14px;
  }}

  li > ul li::before {{
    width: 4px;
    height: 4px;
    top: 13px;
    background: var(--gray-400);
  }}

  /* ── Priority cards within reorder section ── */
  .section--reorder .section-body > p {{
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid var(--amber);
  }}

  .section--reorder .section-body > p:first-of-type {{
    border-left-color: var(--red);
  }}

  .section--reorder .section-body > p:nth-of-type(2) {{
    border-left-color: var(--orange);
  }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 13px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--gray-200);
  }}

  th {{
    background: var(--navy);
    color: #fff;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}

  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--gray-100);
  }}

  tr:nth-child(even) td {{
    background: var(--gray-50);
  }}

  tr:last-child td {{
    border-bottom: none;
  }}

  td.num, th.num {{
    text-align: right;
    font-feature-settings: "tnum";
    font-family: var(--mono);
    font-size: 13px;
  }}

  /* ── Code / preformatted ── */
  code {{
    font-family: var(--mono);
    font-size: 13px;
    background: var(--gray-100);
    padding: 2px 6px;
    border-radius: 4px;
  }}

  pre {{
    background: var(--gray-100);
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 13px;
    margin: 12px 0;
  }}

  /* ── Blockquotes ── */
  blockquote {{
    border-left: 4px solid var(--amber);
    background: var(--amber-dim);
    padding: 14px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
    font-style: normal;
  }}

  blockquote p {{ margin-bottom: 0; }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding-top: 28px;
    margin-top: 12px;
    font-size: 12px;
    color: var(--gray-400);
  }}

  footer .engine {{
    font-weight: 700;
    color: var(--gray-500);
    font-size: 13px;
    letter-spacing: 0.02em;
  }}

  footer .sub {{
    margin-top: 4px;
  }}

  /* ── Print ── */
  @media print {{
    body {{
      background: #fff;
      font-size: 12px;
    }}

    .container {{
      max-width: 100%;
      padding: 0;
    }}

    header {{
      border-radius: 0;
      box-shadow: none;
    }}

    .section {{
      box-shadow: none;
      border: 1px solid var(--gray-200);
      break-inside: avoid;
      page-break-inside: avoid;
      border-radius: 0;
    }}

    footer {{
      margin-top: 24px;
    }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand">Manukora</div>
    <h1>S&amp;OP Monthly Briefing — {report_month}</h1>
    <div class="meta">
      <span>Generated {generation_date}</span>
      <span>Data: {data_source}</span>
      <span>Powered by Claude + pre-computed analytics</span>
    </div>
  </header>

  {body}

  <footer>
    <div class="engine">Manukora S&amp;OP Briefing Engine</div>
    <div class="sub">Pre-computed analytics + Claude inference</div>
  </footer>
</div>
</body>
</html>"""

# Section metadata for visual differentiation
SECTION_META = {
    "Executive Summary":       ("summary", "&#9889;"),
    "Sales Performance":       ("sales",   "&#9650;"),
    "Stock Cover Risk Assessment": ("risk", "&#9888;"),
    "Reorder Recommendations": ("reorder", "&#10132;"),
    "Trend Conflicts & Watch List": ("watch", "&#9673;"),
}


def _normalize_markdown(md_text: str) -> str:
    """Fix common LLM markdown quirks so the markdown lib parses lists correctly."""
    import re
    lines = md_text.split("\n")
    out = []

    def _is_list_line(s: str) -> bool:
        s = s.lstrip()
        if s.startswith("- ") or s.startswith("* ") or s.startswith("\u2022 "):
            return True
        return bool(re.match(r"\d+\.\s", s))

    prev_was_list = False
    for line in lines:
        stripped = line.lstrip()

        # Convert unicode bullet (•) to markdown list syntax
        if stripped.startswith("\u2022 "):
            line = "- " + stripped[2:]
            stripped = line.lstrip()

        is_list = _is_list_line(stripped)

        # Ensure blank line before the start of a list run
        if is_list and not prev_was_list and out and out[-1].strip() != "":
            out.append("")

        # Ensure blank line after a list run ends
        if not is_list and prev_was_list and stripped and out and out[-1].strip() != "":
            out.append("")

        out.append(line)
        prev_was_list = is_list

    return "\n".join(out)


def render_html(markdown_content: str, metadata: dict) -> str:
    """Convert markdown briefing to styled, self-contained HTML."""
    import re

    normalized = _normalize_markdown(markdown_content)
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    body_html = md.convert(normalized)

    # Split on <h2> tags and wrap each in a styled section card
    sections = body_html.split("<h2>")
    wrapped_parts = []
    for i, section in enumerate(sections):
        if i == 0:
            if section.strip():
                wrapped_parts.append(f'<div class="section">{section}</div>')
            continue

        # Extract section title for matching
        title_match = re.match(r"([^<]+)</h2>", section)
        title = title_match.group(1).strip() if title_match else ""
        variant, icon = SECTION_META.get(title, ("default", "&#8226;"))
        rest = section[title_match.end():] if title_match else section

        card = f'''<div class="section section--{variant}">
  <div class="section-header">
    <div class="section-icon">{icon}</div>
    <h2>{title}</h2>
  </div>
  <div class="section-body">{rest}</div>
</div>'''
        wrapped_parts.append(card)

    wrapped_body = "\n".join(wrapped_parts)

    return HTML_TEMPLATE.format(
        report_month=metadata.get("report_month", ""),
        generation_date=metadata.get("generation_date", ""),
        data_source=metadata.get("data_source", ""),
        body=wrapped_body,
    )


# ─────────────────────────────────────────────
# Task 5 — CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manukora S&OP Briefing Engine — generate executive inventory briefings",
    )
    parser.add_argument("--data", required=True, help="Path to inventory CSV")
    parser.add_argument("--output", default="output/", help="Output directory (default: output/)")
    parser.add_argument("--format", choices=["html", "md", "both"], default="both", help="Output format (default: both)")
    parser.add_argument("--month", default="March 2026", help="Report month for briefing header")
    parser.add_argument("--model", default="claude-opus-4-20250514", help="Claude model to use")
    args = parser.parse_args()

    # Ensure output dir exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline ──
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    print(f"  {len(df)} SKUs loaded")

    print("Computing metrics...")
    enriched = compute_metrics(df)

    print(f"Generating briefing via {args.model}...")
    briefing_md = generate_briefing(enriched, args.month, model=args.model)

    # ── Write outputs ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_source = Path(args.data).name

    if args.format in ("md", "both"):
        md_path = output_dir / f"briefing_{timestamp}.md"
        md_path.write_text(briefing_md, encoding="utf-8")
        print(f"  Markdown: {md_path}")

    if args.format in ("html", "both"):
        metadata = {
            "report_month": args.month,
            "generation_date": datetime.now().strftime("%B %d, %Y"),
            "data_source": data_source,
        }
        html_content = render_html(briefing_md, metadata)
        html_path = output_dir / f"briefing_{timestamp}.html"
        html_path.write_text(html_content, encoding="utf-8")
        print(f"  HTML: {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()
