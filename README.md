# Stock Market Analyzer — Yahoo Finance Screener v11

This version adds a **market-style Growth + Financial Health + Momentum Screener** while retaining the editable watchlist and Company Intelligence dashboard.

## What the new Screener does

- Price band: ₹20–₹200 by default
- Market cap: ₹500Cr+ by default
- Liquidity: minimum 20-day average daily traded value threshold
- 3-year sales CAGR and 3-year profit CAGR
- Latest quarterly sales/profit YoY growth
- Debt/equity, debt trend and interest coverage
- ROE, derived ROCE, operating margin and net margin
- Operating cash flow, free cash flow and OCF/net-profit relationship
- Receivables and inventory growth flags where Yahoo provides the rows
- 1M / 3M / 6M price returns
- 50 DMA / 200 DMA relationship, RSI and volume vs 20-day average
- P/E, P/B, EV/EBITDA and PEG where Yahoo provides them
- Yahoo institutional/insider ownership where available
- Explicit missing-data handling: missing fields stay `—`
- Rule-based red flags, including >50% 1-month price gain as a **momentum flag only**
- Transparent 100-point score with the requested weights
- Coverage-adjusted score: unavailable components are excluded from the denominator instead of being silently scored as zero
- Categories:
  - A — Strong fundamentals + strong momentum
  - B — Strong fundamentals + emerging momentum
  - C — High momentum but fundamental risks / insufficient evidence
- Top 15–25 research table and a 5-company deep-research shortlist
- Company-by-company research view with growth, debt, cash flow, valuation, momentum, risks and next-quarter monitoring points

## Scoring weights

| Component | Weight |
|---|---:|
| Revenue growth | 15 |
| Profit growth / acceleration | 15 |
| Debt / balance sheet | 15 |
| ROCE / ROE / profitability | 15 |
| Cash-flow quality | 10 |
| Recent quarterly performance | 10 |
| Price / volume momentum | 10 |
| Valuation | 5 |
| Ownership / institutions | 5 |
| **Total** | **100** |

The score is an analytical screening aid, not a prediction or BUY/SELL recommendation.

## Universe source vs financial-data source

**Yahoo Finance remains the source for financial and market metrics.**

Yahoo Finance does not provide a dependable, complete NSE+BSE security universe for a custom Indian-market screener. Therefore the app supports uploading an NSE/BSE symbol-master CSV with:

```text
Company,Symbol,Exchange
Example Ltd,EXAMPLE,NSE
Example Co,500001,BSE
```

The exchange CSV is used only to enumerate securities and construct Yahoo tickers (`.NS` / `.BO`). Financial and market metrics are fetched from Yahoo Finance.

The app also lets you scan the current watchlist immediately for testing.

## Important Yahoo limitations

Yahoo Finance does not reliably expose all Indian-market due-diligence fields requested by the screener. In particular, promoter pledge, promoter-holding history, industry-average P/E, historical P/E, auditor qualifications, related-party transactions, contingent liabilities, detailed order books and all corporate/legal filings may be unavailable. These are **not inferred** and remain `—`.

For those fields, the app directs you to company/NSE/BSE filings rather than manufacturing an answer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cloud

Deploy `app.py` from GitHub on Streamlit Community Cloud. Configure `SUPABASE_URL` and `SUPABASE_KEY` as Streamlit secrets for persistent cloud watchlists and create the table using `supabase_schema.sql`.
