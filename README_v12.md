# Stock Market Analyzer — Yahoo Finance Growth Screener v12

The growth screener no longer asks the user to upload a stock-universe CSV.

## Automatic universe
- NSE mainboard equity securities are discovered from the current NSE equity master.
- BSE active equity securities are attempted automatically through the BSE equity list endpoint.
- NSE SME is optional and OFF by default because the screen is intended to avoid SME/illiquid characteristics.
- Exchange masters are used only to enumerate securities. Yahoo Finance remains the source for price, volume, financial statements, valuation and technical metrics.

## Screening flow
1. Fetch NSE/BSE universe from the sidebar.
2. Yahoo Finance batch price/volume history pre-filters by price and average 20-day traded value.
3. Only the resulting candidates receive the more expensive per-company financial analysis.
4. Fundamental, balance-sheet, cash-flow, quarterly, valuation and momentum criteria are calculated.
5. Results are displayed in the main area, with categories A/B/C and the transparent 100-point framework.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
