# Growth Screener v15 — empty-result fix

## Root cause fixed
The fast Yahoo prefilter used the columns:
- `Market Price`
- `Avg 20D Turnover ₹Cr`

The final qualification function expected:
- `Price`
- `Avg Daily Turnover 20D ₹Cr`

As a result, the app could correctly report that 100 securities passed the fast Yahoo price/liquidity scan, analyze all 100, and then discard every row at the final filter because the expected column names were missing.

## Additional fixes
- Final filter now accepts both naming conventions.
- Fast-scan values are preserved as aliases.
- Data Status is displayed in the screener table.
- Strict fundamental gate is coverage-aware: at least 70% of the 11 fundamental criteria must have data, and every available criterion must pass. Missing Yahoo data is not silently treated as a failure.
- If strict mode still returns zero, the app explicitly tells the user how many stocks passed the market/liquidity stage.
- Added `START_GROWTH_SCREENER_LOCALHOST.bat` for direct Windows localhost launch.

## v16 performance + crash fix
- Fixed `NameError: math is not defined` by importing `math`.
- Reduced default expensive fundamental analysis to 40 candidates.
- Fast market universe scan remains batched and only uses 1 month of history.
- Added one batched 1-year technical-history request for the shortlisted liquid candidates.
- Removed the per-candidate 3-year price-history request from the fundamental worker.
- Removed the expensive `Ticker.info` call from the bulk fundamental stage; margins and ROE are derived from financial statements, while valuation fields remain unavailable in the bulk scan rather than being fabricated.
- Reused price and turnover from the fast market scan.
- Increased fundamental worker pool to up to 8 workers.
- Added `lru_cache` to reuse completed candidate analyses across Streamlit reruns.
- The strict fundamental gate remains coverage-aware and now executes without the missing `math` import error.

## v17 — Empty Screener Result Fix

- Fixed the main empty-results failure in the growth screener.
- The basic market filter requires Market Cap >= configured threshold. v16 relied on `Ticker.fast_info['market_cap']` during the bulk pass; for some NSE/BSE symbols this field can be unavailable.
- v17 now falls back to Yahoo financial-statement share count and derives market cap as `price × shares outstanding`, converted to ₹ crore. Missing share count remains missing; no value is fabricated.
- Added `Market Cap Source` so the user can see whether the value came from Yahoo fast_info or was derived from Yahoo share count.
- Added market-gate diagnostics showing price pass, turnover pass, market-cap pass, and market-cap data coverage before the final table.
- Added an explicit warning when the market-qualified set is empty, distinguishing data coverage from a genuinely empty result.
- Kept the fast batch price/volume scan and limited fundamental analyses to the configured maximum.


## v18 Market-cap gate fix
- Market cap is now fetched in a separate stage before fundamental analysis.
- Uses Yahoo `fast_info` when available, then Yahoo `get_shares()` × current price as fallback.
- Minimum market-cap filter is applied before expensive financial-statement calls.
- Fast liquid candidate pool default increased to 300; fundamental analysis remains capped separately (default 50).
- Added explicit diagnostics for market-cap availability and market-cap-qualified count.
- A stock below the configured minimum market cap is correctly excluded; e.g. ₹1,842 Cr does not qualify against a ₹5,000 Cr minimum.
