# v11 — Growth + Financial Health + Momentum Screener

Added a Screener tab with:
- NSE/BSE universe CSV upload support
- ₹20–₹200 price filter
- market-cap and liquidity filters
- annual and quarterly growth metrics
- debt/balance-sheet metrics
- ROE/ROCE/profitability metrics
- cash-flow quality metrics
- momentum and volume metrics
- valuation metrics
- transparent weighted score / 100
- score coverage percentage for missing Yahoo data
- A/B/C research categories
- top research table and five-company deep-research shortlist
- detailed per-company screening view
- explicit Yahoo data limitations for promoter pledge, industry P/E, historical P/E and filing-level corporate governance data

## v12 requested enhancement
- Removed the NSE/BSE universe CSV upload requirement from the Streamlit screener.
- Added automatic NSE equity universe discovery and optional BSE active-equity discovery.
- Added fast Yahoo price/liquidity pre-filtering across the discovered universe before expensive fundamental analysis, so the scan is not limited to the first alphabetically listed securities.
- Moved all screener controls into the Streamlit sidebar.
- Screener results remain in the main content area for readability.
