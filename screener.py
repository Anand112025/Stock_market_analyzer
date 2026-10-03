import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import io
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Yahoo Finance is the financial/market data source.  The app may use an NSE/BSE
# symbol master only to enumerate securities; it does not use a third-party
# fundamental database for the metrics below.



NSE_EQUITY_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
NSE_SME_URL = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
BSE_LIST_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"


def _web_headers(referer="https://www.nseindia.com/"):
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153.0 Safari/537.36",
        "Accept": "text/csv,application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }


def fetch_market_universe(include_sme=False, include_bse=True):
    """Fetch a fresh NSE/BSE equity universe; no user upload is required.

    Exchange masters are used only for security enumeration. All screening
    fundamentals, prices and technicals still come from Yahoo Finance.
    """
    rows = []
    errors = []
    session = requests.Session()
    session.headers.update(_web_headers())

    # NSE mainboard equity master
    nse = pd.DataFrame()
    for url in NSE_EQUITY_URLS:
        try:
            session.get("https://www.nseindia.com", timeout=15)
            r = session.get(url, timeout=30)
            r.raise_for_status()
            nse = pd.read_csv(io.BytesIO(r.content))
            if "SYMBOL" in nse.columns:
                break
        except Exception as exc:
            errors.append(f"NSE equity list: {exc}")
    if not nse.empty:
        # Some NSE downloads omit the SERIES column.  DataFrame.get() would
        # then return a plain string, and calling .isin() on that string
        # raises: AttributeError: 'str' object has no attribute 'isin'.
        # Treat a missing SERIES column as mainboard EQ instead.
        if "SERIES" in nse.columns:
            series = nse["SERIES"].astype(str).str.strip().str.upper()
            nse = nse[series.isin(["EQ", "BE"])].copy()
        else:
            nse = nse.copy()
        for _, r in nse.iterrows():
            sym = str(r.get("SYMBOL", "")).strip().upper()
            name = str(r.get("NAME OF COMPANY", sym)).strip()
            if sym:
                rows.append({"Company": name, "Symbol": f"{sym}.NS", "Exchange": "NSE", "Segment": "Mainboard"})

    # Optional SME universe. Off by default because the user's base objective
    # explicitly wants to avoid SME/illiquid characteristics.
    if include_sme:
        try:
            r = session.get(NSE_SME_URL, timeout=30)
            r.raise_for_status()
            sme = pd.read_csv(io.BytesIO(r.content))
            for _, rr in sme.iterrows():
                sym = str(rr.get("SYMBOL", "")).strip().upper()
                name = str(rr.get("NAME OF COMPANY", sym)).strip()
                if sym:
                    rows.append({"Company": name, "Symbol": f"{sym}.NS", "Exchange": "NSE", "Segment": "SME"})
        except Exception as exc:
            errors.append(f"NSE SME list: {exc}")

    # BSE active equity master. BSE's endpoint can occasionally reject cloud
    # requests; the screener therefore continues with NSE if BSE is unavailable.
    if include_bse:
        try:
            params = {"scripcode": "", "Group": "", "industry": "", "segment": "Equity", "status": "Active"}
            r = session.get(BSE_LIST_URL, params=params, headers=_web_headers("https://www.bseindia.com/"), timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("Table", data.get("Table1", []))
            if isinstance(data, list):
                for rr in data:
                    code = str(rr.get("SCRIP_CD", rr.get("SCRIPCODE", rr.get("Scripcode", "")))).strip()
                    name = str(rr.get("LONG_NAME", rr.get("NAME", rr.get("SCRIP_NAME", "")))).strip()
                    if code.isdigit() and name:
                        rows.append({"Company": name, "Symbol": f"{code}.BO", "Exchange": "BSE", "Segment": "Mainboard/Equity"})
        except Exception as exc:
            errors.append(f"BSE equity list: {exc}")

    universe = pd.DataFrame(rows)
    if universe.empty:
        return universe, errors
    universe = universe.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    return universe, errors

WEIGHTS = {
    "Revenue Growth": 15,
    "Profit Growth": 15,
    "Debt / Balance Sheet": 15,
    "ROCE / ROE / Profitability": 15,
    "Cash Flow Quality": 10,
    "Recent Quarterly Performance": 10,
    "Price / Volume Momentum": 10,
    "Valuation": 5,
    "Ownership / Institutions": 5,
}


def num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def pct(a, b):
    a, b = num(a), num(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1) * 100


def row(frame, names):
    if frame is None or frame.empty:
        return None
    for n in names:
        if n in frame.index:
            return frame.loc[n]
    return None


def rowval(series, col):
    if series is None:
        return np.nan
    try:
        return num(series[col])
    except Exception:
        return np.nan


def cols_desc(frame):
    if frame is None or frame.empty:
        return []
    out = []
    for c in frame.columns:
        try:
            d = pd.to_datetime(c)
            if pd.notna(d):
                out.append((d, c))
        except Exception:
            pass
    return sorted(out, reverse=True)


def cagr(latest, oldest, years=3):
    latest, oldest = num(latest), num(oldest)
    if pd.isna(latest) or pd.isna(oldest) or latest <= 0 or oldest <= 0:
        return np.nan
    return ((latest / oldest) ** (1 / years) - 1) * 100


def rsi(close, period=14):
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < period + 1:
        return np.nan
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    if pd.isna(loss.iloc[-1]) or pd.isna(gain.iloc[-1]):
        return np.nan
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100 - (100 / (1 + rs))


def latest_yoy(series, qcols):
    if series is None or len(qcols) < 4:
        return np.nan
    latest = rowval(series, qcols[0][1])
    # Four quarters back may not be in the first five columns on some Yahoo feeds.
    for d, c in qcols[1:]:
        if abs((qcols[0][0] - d).days - 365) <= 100:
            return pct(latest, rowval(series, c))
    return np.nan


def _holders(ticker):
    out = {"Insider Holding %": np.nan, "Institution Holding %": np.nan, "Mutual Fund Holding %": np.nan}
    try:
        mh = ticker.major_holders
        if isinstance(mh, pd.DataFrame) and not mh.empty:
            for _, r in mh.iterrows():
                label = str(r.iloc[1]).lower() if len(r) > 1 else str(r.iloc[0]).lower()
                value = num(r.iloc[0]) if len(r) > 1 else np.nan
                if "insider" in label:
                    out["Insider Holding %"] = value * 100 if pd.notna(value) and value <= 1 else value
                elif "institution" in label:
                    out["Institution Holding %"] = value * 100 if pd.notna(value) and value <= 1 else value
    except Exception:
        pass
    try:
        ih = ticker.institutional_holders
        if isinstance(ih, pd.DataFrame) and not ih.empty:
            # This is institutional ownership, not promoter ownership.
            if "% Out" in ih.columns:
                vals = pd.to_numeric(ih["% Out"], errors="coerce").dropna()
                if not vals.empty:
                    out["Institution Holding %"] = vals.sum() * 100 if vals.max() <= 1 else vals.sum()
    except Exception:
        pass
    return out


def _score_range(value, low, high):
    value = num(value)
    if pd.isna(value):
        return np.nan
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def score_candidate(d):
    """Transparent component scoring. Missing components are excluded from the
    denominator instead of being treated as failures or successes."""
    components = {}

    # 15 Revenue growth
    rg = d.get("3Y Sales CAGR %")
    latest_sales = d.get("Latest Quarterly Sales YoY %")
    vals = [v for v in [rg, latest_sales] if pd.notna(num(v))]
    components["Revenue Growth"] = np.mean([_score_range(v, 0, 25) for v in vals]) if vals else np.nan

    # 15 Profit growth / acceleration
    pg = d.get("3Y Profit CAGR %")
    latest_profit = d.get("Latest Quarterly Profit YoY %")
    vals = [v for v in [pg, latest_profit] if pd.notna(num(v))]
    components["Profit Growth"] = np.mean([_score_range(v, 0, 35) for v in vals]) if vals else np.nan

    # 15 balance sheet
    de = d.get("Debt / Equity")
    debt_yoy = d.get("Debt YoY %")
    ic = d.get("Interest Coverage")
    vals = []
    if pd.notna(num(de)): vals.append(max(0, min(1, (1.0 - de) / 1.0)))
    if pd.notna(num(debt_yoy)): vals.append(1.0 if debt_yoy <= -10 else 0.75 if debt_yoy < 0 else 0.45 if debt_yoy <= 15 else 0.1)
    if pd.notna(num(ic)): vals.append(_score_range(ic, 1.5, 6.0))
    components["Debt / Balance Sheet"] = np.mean(vals) if vals else np.nan

    # 15 profitability
    roe, roce, opm, npm = [d.get(k) for k in ["ROE %", "ROCE %", "Operating Margin %", "Net Margin %"]]
    vals = []
    for v in [roe, roce]:
        if pd.notna(num(v)): vals.append(_score_range(v, 0, 25))
    for v in [opm, npm]:
        if pd.notna(num(v)): vals.append(_score_range(v, 0, 15))
    components["ROCE / ROE / Profitability"] = np.mean(vals) if vals else np.nan

    # 10 cash flow
    ocf = d.get("Operating Cash Flow")
    fcf = d.get("Free Cash Flow")
    ocf_np = d.get("OCF / Net Profit %")
    vals = []
    if pd.notna(num(ocf)): vals.append(1 if ocf > 0 else 0)
    if pd.notna(num(fcf)): vals.append(1 if fcf > 0 else 0)
    if pd.notna(num(ocf_np)): vals.append(_score_range(ocf_np, 0, 100))
    components["Cash Flow Quality"] = np.mean(vals) if vals else np.nan

    # 10 recent quarterly performance
    vals = [v for v in [d.get("Latest Quarterly Sales YoY %"), d.get("Latest Quarterly Profit YoY %")] if pd.notna(num(v))]
    components["Recent Quarterly Performance"] = np.mean([_score_range(v, 0, 30) for v in vals]) if vals else np.nan

    # 10 momentum
    vals = []
    for k in ["1M Return %", "3M Return %", "6M Return %"]:
        v = num(d.get(k))
        if pd.notna(v): vals.append(_score_range(v, -10, 50))
    if pd.notna(num(d.get("Price vs 50 DMA %"))): vals.append(1 if d["Price vs 50 DMA %"] > 0 else 0)
    if pd.notna(num(d.get("50 DMA vs 200 DMA %"))): vals.append(1 if d["50 DMA vs 200 DMA %"] > 0 else 0)
    if pd.notna(num(d.get("Volume vs 20D Avg %"))): vals.append(_score_range(d["Volume vs 20D Avg %"], -50, 100))
    components["Price / Volume Momentum"] = np.mean(vals) if vals else np.nan

    # 5 valuation: relative to growth, without hard-rejecting P/E > 20/30.
    pe = d.get("P/E")
    peg = d.get("PEG")
    growth = d.get("3Y Profit CAGR %")
    vals = []
    if pd.notna(num(peg)): vals.append(1 if peg <= 1 else 0.75 if peg <= 1.5 else 0.4 if peg <= 2.5 else 0.1)
    elif pd.notna(num(pe)) and pd.notna(num(growth)) and growth > 0:
        ratio = pe / growth
        vals.append(1 if ratio <= 0.75 else 0.75 if ratio <= 1.25 else 0.45 if ratio <= 2 else 0.15)
    elif pd.notna(num(pe)):
        vals.append(1 if pe <= 15 else 0.75 if pe <= 25 else 0.45 if pe <= 40 else 0.15)
    components["Valuation"] = np.mean(vals) if vals else np.nan

    # 5 ownership/institutions — Yahoo does not reliably expose promoter/pledge.
    vals = []
    if pd.notna(num(d.get("Institution Holding %"))): vals.append(_score_range(d["Institution Holding %"], 0, 25))
    if pd.notna(num(d.get("Insider Holding %"))): vals.append(_score_range(d["Insider Holding %"], 0, 50))
    components["Ownership / Institutions"] = np.mean(vals) if vals else np.nan

    weighted = 0.0
    available = 0.0
    for name, weight in WEIGHTS.items():
        v = components.get(name, np.nan)
        if pd.notna(v):
            weighted += float(v) * weight
            available += weight
    normalized = weighted / available * 100 if available else np.nan
    return components, weighted, available, normalized


def prefilter_market_universe(universe, min_price=20.0, max_price=200.0, min_turnover_cr=0.25, max_candidates=1000):
    """Fast market-data prefilter before expensive per-company financial calls.

    The exchange universe can contain thousands of securities.  The old
    implementation made many small Yahoo requests and could appear to hang.
    This version deliberately uses larger batches and only one month of
    history. It performs a lightweight market scan first and reserves all
    longer-history/financial-statement work for the resulting liquid candidates.

    ``max_candidates`` limits the expensive fundamental stage, not the
    exchange-universe discovery stage.  Candidates are selected by average
    20-day traded value so the scan does not simply take the first symbols.
    """
    if universe is None or universe.empty:
        return pd.DataFrame()

    u = universe.copy()
    if "Symbol" not in u.columns:
        return pd.DataFrame()

    symbols = u["Symbol"].astype(str).str.upper().str.strip().drop_duplicates().tolist()
    matches = []

    # Larger chunks dramatically reduce request overhead for a 2,000–3,000
    # security universe.  Yahoo can occasionally reject a very large batch, so
    # a failed batch is retried once at half size.
    primary_chunk = 250
    for i in range(0, len(symbols), primary_chunk):
        chunk = symbols[i:i + primary_chunk]
        data = pd.DataFrame()
        for attempt_chunk in [chunk, chunk[:max(50, len(chunk)//2)]]:
            try:
                data = yf.download(
                    attempt_chunk,
                    period="1mo",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                    timeout=15,
                )
                if data is not None and not data.empty:
                    # If the reduced retry was used, only those symbols are
                    # available in this response.
                    chunk = attempt_chunk
                    break
            except Exception:
                data = pd.DataFrame()
        if data is None or data.empty:
            continue

        multi = isinstance(data.columns, pd.MultiIndex)
        for sym in chunk:
            try:
                if multi:
                    top = data.columns.get_level_values(0)
                    if sym not in top:
                        continue
                    h = data[sym]
                else:
                    h = data

                if h is None or h.empty or "Close" not in h.columns or "Volume" not in h.columns:
                    continue
                close = pd.to_numeric(h["Close"], errors="coerce")
                vol = pd.to_numeric(h["Volume"], errors="coerce")
                valid = close.notna() & vol.notna()
                close = close[valid]
                vol = vol[valid]
                if close.empty:
                    continue

                price = float(close.iloc[-1])
                turnover = (close.tail(20) * vol.tail(20)).mean() / 1e7
                turnover = float(turnover) if pd.notna(turnover) else np.nan
                if (
                    min_price <= price <= max_price
                    and pd.notna(turnover)
                    and turnover >= min_turnover_cr
                ):
                    matches.append({
                        "Symbol": sym,
                        "Market Price": price,
                        "Avg 20D Turnover ₹Cr": turnover,
                    })
            except Exception:
                continue

    if not matches:
        return pd.DataFrame()

    m = pd.DataFrame(matches).drop_duplicates("Symbol")
    out = u.merge(m, on="Symbol", how="inner")
    out = (
        out.sort_values("Avg 20D Turnover ₹Cr", ascending=False, na_position="last")
        .head(int(max_candidates))
        .reset_index(drop=True)
    )
    return out



def enrich_market_caps(candidates, max_workers=12):
    """Fetch market-cap data for shortlisted liquid candidates before fundamentals.

    Yahoo's fast_info market_cap is not consistently populated for Indian
    NSE/BSE symbols.  For those cases use Yahoo's shares history and multiply
    the latest share count by the already-known market price.  This stage is
    intentionally separate from the fundamental analysis so the minimum
    market-cap filter is applied *before* expensive financial-statement calls.
    """
    if candidates is None or candidates.empty or "Symbol" not in candidates.columns:
        return pd.DataFrame() if candidates is None else candidates.copy()

    out = candidates.copy()
    if "Market Cap ₹Cr" not in out.columns:
        out["Market Cap ₹Cr"] = np.nan
    if "Market Cap Source" not in out.columns:
        out["Market Cap Source"] = "Unavailable"

    def one(row_tuple):
        company, symbol = row_tuple
        price = num(company.get("Market Price", np.nan)) if isinstance(company, dict) else np.nan
        if pd.isna(price):
            return symbol, np.nan, "Unavailable"
        try:
            t = yf.Ticker(symbol)
            # Fast path where Yahoo exposes market cap.
            try:
                fi = t.fast_info
                mc = num(fi.get("market_cap")) if hasattr(fi, "get") else np.nan
                if pd.notna(mc) and mc > 0:
                    return symbol, mc / 1e7, "Yahoo fast_info"
            except Exception:
                pass

            # Robust fallback: Yahoo shares history. Prefer the latest value.
            try:
                sh = t.get_shares()
                if isinstance(sh, pd.DataFrame) and not sh.empty:
                    numeric = sh.apply(pd.to_numeric, errors="coerce")
                    vals = numeric.stack().dropna()
                    if not vals.empty:
                        shares = float(vals.iloc[-1])
                        if shares > 0:
                            return symbol, price * shares / 1e7, "Derived: price × Yahoo shares"
                elif isinstance(sh, pd.Series) and not sh.empty:
                    vals = pd.to_numeric(sh, errors="coerce").dropna()
                    if not vals.empty:
                        shares = float(vals.iloc[-1])
                        if shares > 0:
                            return symbol, price * shares / 1e7, "Derived: price × Yahoo shares"
            except Exception:
                pass
        except Exception:
            pass
        return symbol, np.nan, "Unavailable"

    rows = []
    records = out.to_dict("records")
    workers = min(int(max_workers), max(2, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, r) for r in records]
        for f in as_completed(futures):
            try:
                rows.append(f.result())
            except Exception:
                pass

    cap_df = pd.DataFrame(rows, columns=["Symbol", "Market Cap ₹Cr", "Market Cap Source"])
    if cap_df.empty:
        return out
    cap_df = cap_df.drop_duplicates("Symbol")
    out = out.drop(columns=["Market Cap ₹Cr", "Market Cap Source"], errors="ignore")
    return out.merge(cap_df, on="Symbol", how="left")

def enrich_market_technicals(candidates):
    """Fetch one batched 1-year history for the already-selected liquid candidates."""
    if candidates is None or candidates.empty or "Symbol" not in candidates.columns:
        return pd.DataFrame()
    symbols = candidates["Symbol"].astype(str).str.upper().str.strip().drop_duplicates().tolist()
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(
            symbols, period="1y", interval="1d", auto_adjust=False, actions=False,
            progress=False, group_by="ticker", threads=True, timeout=20
        )
    except Exception:
        return candidates.copy()
    if data is None or data.empty:
        return candidates.copy()
    rows = []
    multi = isinstance(data.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            h = data[sym] if multi and sym in data.columns.get_level_values(0) else data
            if h is None or h.empty or "Close" not in h.columns:
                continue
            close = pd.to_numeric(h["Close"], errors="coerce").dropna()
            vol = pd.to_numeric(h.get("Volume", pd.Series(dtype=float)), errors="coerce").reindex(close.index)
            if close.empty:
                continue
            price = float(close.iloc[-1])
            def hist_return(days):
                target = close.index[-1] - pd.Timedelta(days=days)
                prior = close.loc[:target]
                return pct(price, prior.iloc[-1]) if not prior.empty else np.nan
            dma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            dma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            valid_vol = vol.dropna()
            avg20 = valid_vol.tail(20).mean() if len(valid_vol) >= 20 else np.nan
            rows.append({
                "Symbol": sym,
                "1M Return %": hist_return(30),
                "3M Return %": hist_return(90),
                "6M Return %": hist_return(180),
                "50 DMA": dma50,
                "200 DMA": dma200,
                "Price vs 50 DMA %": pct(price, dma50),
                "Price vs 200 DMA %": pct(price, dma200),
                "50 DMA vs 200 DMA %": pct(dma50, dma200),
                "RSI 14": rsi(close),
                "Volume vs 20D Avg %": (valid_vol.iloc[-1] / avg20 - 1) * 100 if pd.notna(avg20) and avg20 else np.nan,
            })
        except Exception:
            continue
    if not rows:
        return candidates.copy()
    return candidates.merge(pd.DataFrame(rows), on="Symbol", how="left")


@lru_cache(maxsize=500)
def analyze_candidate(company, symbol, exchange="", market_price=np.nan, market_turnover=np.nan):
    d = {"Company": company, "Symbol": symbol, "Exchange": exchange, "Data Status": "STARTING"}
    try:
        t = yf.Ticker(symbol)
        # Avoid the expensive quoteSummary/info call during the bulk screen.
        # Market cap/valuation fields are not part of the strict fundamental
        # gate and can be enriched later for the small final shortlist.
        info = {}
        # Do not depend on Ticker.fast_info for market cap during the bulk
        # fundamental pass.  For Indian NSE/BSE symbols this field can be
        # unavailable even when the financial statements contain share count.
        # We therefore derive market cap later as price * shares outstanding.
        try:
            fast = t.fast_info
            if hasattr(fast, "get"):
                info["marketCap"] = fast.get("market_cap", np.nan)
        except Exception:
            fast = {}
        # The fast universe scan already fetched price and turnover in batches.
        # Reuse those values instead of downloading a second 3-year history for
        # every candidate; this is the main speed improvement in v16.
        price = num(market_price)
        if pd.notna(price):
            d["Price"] = price
        d["Avg Daily Turnover 20D ₹Cr"] = num(market_turnover)
        # Preserve the fast prefilter field names as aliases so the final
        # qualification stage can never discard every row just because the
        # lightweight scan and fundamental stage used different column names.
        d["Market Price"] = d.get("Price", np.nan)
        d["Avg 20D Turnover ₹Cr"] = d.get("Avg Daily Turnover 20D ₹Cr", np.nan)
        # Prefer Yahoo fast_info when available; otherwise derive market cap
        # from the latest balance-sheet share count and the already-fetched
        # market price. This avoids dropping otherwise valid Indian companies
        # merely because fast_info did not expose market_cap.
        fast_mcap = num(info.get("marketCap"))
        d["Market Cap ₹Cr"] = fast_mcap / 1e7 if pd.notna(fast_mcap) else np.nan
        d["P/E"] = np.nan
        d["P/B"] = np.nan
        d["EV / EBITDA"] = np.nan
        d["PEG"] = np.nan
        d["Forward P/E"] = np.nan
        d["Industry"] = ""
        d["Sector"] = ""

        income = getattr(t, "income_stmt", pd.DataFrame())
        icols = cols_desc(income)
        rev = row(income, ["Total Revenue","Operating Revenue","Revenue"])
        ni = row(income, ["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"])
        op = row(income, ["Operating Income","Operating Income Or Loss"])
        ebit = op
        da = row(income, ["Depreciation And Amortization","Reconciled Depreciation","Depreciation"])
        for i,(dt,col) in enumerate(icols[:4]):
            d[f"Revenue FY{i}"] = rowval(rev,col)
            d[f"Profit FY{i}"] = rowval(ni,col)
        if len(icols)>=4:
            d["3Y Sales CAGR %"] = cagr(d.get("Revenue FY0"), d.get("Revenue FY3"), 3)
            d["3Y Profit CAGR %"] = cagr(d.get("Profit FY0"), d.get("Profit FY3"), 3)
        bs = getattr(t, "balance_sheet", pd.DataFrame())
        bcols = cols_desc(bs)
        debt = row(bs,["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt"])
        equity = row(bs,["Stockholders Equity","Common Stock Equity","Total Equity Gross Minority Interest"])
        assets = row(bs,["Total Assets"])
        shares = row(bs,["Ordinary Shares Number","Share Issued","Common Stock Shares Outstanding","Common Shares Outstanding"])
        curr_liab = row(bs,["Current Liabilities"])
        cash = row(bs,["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents","Cash Financial"])
        receiv = row(bs,["Accounts Receivable","Receivables","Net Receivables"])
        if icols:
            rev_latest = rowval(rev, icols[0][1])
            op_latest = rowval(op, icols[0][1])
            ni_latest = rowval(ni, icols[0][1])
            d["Operating Margin %"] = op_latest / rev_latest * 100 if pd.notna(op_latest) and pd.notna(rev_latest) and rev_latest != 0 else np.nan
            d["Net Margin %"] = ni_latest / rev_latest * 100 if pd.notna(ni_latest) and pd.notna(rev_latest) and rev_latest != 0 else np.nan
        if bcols and icols:
            ni_latest = rowval(ni, icols[0][1])
            eq_latest = rowval(equity, bcols[0][1])
            d["ROE %"] = ni_latest / eq_latest * 100 if pd.notna(ni_latest) and pd.notna(eq_latest) and eq_latest != 0 else np.nan

        # Market-cap fallback: latest price (₹) × latest shares outstanding,
        # converted to ₹ crore. Missing share count remains missing.
        if pd.isna(d.get("Market Cap ₹Cr")) and bcols and pd.notna(d.get("Price")):
            shares_latest = rowval(shares, bcols[0][1])
            if pd.notna(shares_latest) and shares_latest > 0:
                d["Market Cap ₹Cr"] = d["Price"] * shares_latest / 1e7
                d["Market Cap Source"] = "Derived: price × shares outstanding"
            else:
                # Some Indian listings do not expose share count in the
                # balance sheet. Try Yahoo's dedicated shares endpoint.
                try:
                    sh = t.get_shares()
                    vals = sh.apply(pd.to_numeric, errors="coerce").stack().dropna() if isinstance(sh, pd.DataFrame) else pd.to_numeric(sh, errors="coerce").dropna()
                    if len(vals):
                        shares_latest = float(vals.iloc[-1])
                        if shares_latest > 0:
                            d["Market Cap ₹Cr"] = d["Price"] * shares_latest / 1e7
                            d["Market Cap Source"] = "Derived: price × Yahoo shares"
                        else:
                            d["Market Cap Source"] = "Unavailable"
                    else:
                        d["Market Cap Source"] = "Unavailable"
                except Exception:
                    d["Market Cap Source"] = "Unavailable"
        elif pd.notna(d.get("Market Cap ₹Cr")):
            d["Market Cap Source"] = "Yahoo fast_info"
        else:
            d["Market Cap Source"] = "Unavailable"

        inventory = row(bs,["Inventory","Inventories Raw Materials"])
        for i,(dt,col) in enumerate(bcols[:4]):
            d[f"Debt FY{i}"] = rowval(debt,col)
        if len(bcols)>=2:
            d["Debt YoY %"] = pct(d.get("Debt FY0"), d.get("Debt FY1"))
        d["Debt / Equity"] = (d.get("Debt FY0",np.nan)/d.get("Equity FY0",np.nan)) if pd.notna(d.get("Debt FY0")) and pd.notna(d.get("Equity FY0")) and d.get("Equity FY0") else np.nan
        if pd.notna(d.get("Debt FY0")) and pd.notna(d.get("Equity FY0")):
            d["Debt / Equity"] = d["Debt FY0"]/d["Equity FY0"]
        d["Debt Trend"] = "DECREASING" if pd.notna(d.get("Debt YoY %")) and d["Debt YoY %"] < -5 else "INCREASING" if pd.notna(d.get("Debt YoY %")) and d["Debt YoY %"] > 5 else "STABLE / MIXED" if pd.notna(d.get("Debt YoY %")) else "—"
        d["Receivables YoY %"] = pct(rowval(receiv,bcols[0][1]) if bcols else np.nan, rowval(receiv,bcols[1][1]) if len(bcols)>1 else np.nan)
        d["Inventory YoY %"] = pct(rowval(inventory,bcols[0][1]) if bcols else np.nan, rowval(inventory,bcols[1][1]) if len(bcols)>1 else np.nan)

        # ROCE derived from Yahoo statement data when the required rows exist.
        if bcols and icols:
            ebit_latest = rowval(ebit, icols[0][1])
            capital = rowval(assets,bcols[0][1]) - rowval(curr_liab,bcols[0][1])
            d["ROCE %"] = ebit_latest / capital * 100 if pd.notna(ebit_latest) and pd.notna(capital) and capital > 0 else np.nan
        interest = row(income,["Interest Expense Non Operating","Interest Expense"])
        if icols:
            iv = rowval(interest,icols[0][1])
            ev = rowval(ebit,icols[0][1])
            d["Interest Coverage"] = ev / abs(iv) if pd.notna(ev) and pd.notna(iv) and iv != 0 else np.nan

        cf = getattr(t, "cashflow", pd.DataFrame())
        ccols = cols_desc(cf)
        ocf = row(cf,["Operating Cash Flow","Total Cash From Operating Activities"])
        capex = row(cf,["Capital Expenditure","Capital Expenditure Reported"])
        fcf = row(cf,["Free Cash Flow"])
        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf + capex
        d["Operating Cash Flow"] = rowval(ocf,ccols[0][1]) if ccols else np.nan
        d["Free Cash Flow"] = rowval(fcf,ccols[0][1]) if ccols else np.nan
        latest_np = d.get("Profit FY0")
        d["OCF / Net Profit %"] = d["Operating Cash Flow"]/latest_np*100 if pd.notna(d["Operating Cash Flow"]) and pd.notna(latest_np) and latest_np != 0 else np.nan
        d["3Y Cumulative OCF"] = sum(rowval(ocf,c) for _,c in ccols[:3]) if ocf is not None and len(ccols)>=3 else np.nan
        d["3Y Cumulative FCF"] = sum(rowval(fcf,c) for _,c in ccols[:3]) if fcf is not None and len(ccols)>=3 else np.nan

        qi = getattr(t, "quarterly_income_stmt", pd.DataFrame())
        qcols = cols_desc(qi)
        qrev = row(qi,["Total Revenue","Operating Revenue","Revenue"])
        qni = row(qi,["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"])
        if qcols:
            d["Latest Quarter"] = qcols[0][0].strftime("%Y-%m-%d")
            d["Latest Quarterly Sales YoY %"] = latest_yoy(qrev,qcols)
            d["Latest Quarterly Profit YoY %"] = latest_yoy(qni,qcols)
            d["Latest Quarterly Revenue"] = rowval(qrev,qcols[0][1])
            d["Latest Quarterly Profit"] = rowval(qni,qcols[0][1])

        d.update(_holders(t))
        # Yahoo generally does not expose reliable Indian promoter-pledge or promoter-holding history.
        d["Promoter Holding %"] = np.nan
        d["Promoter Pledge %"] = np.nan
        d["Industry P/E"] = np.nan
        d["Historical P/E"] = np.nan

        # Flags / evidence
        flags=[]
        if pd.notna(d.get("1M Return %")) and d["1M Return %"] > 50: flags.append("1M price gain >50% — momentum flag, not fundamental proof")
        if pd.notna(d.get("Debt / Equity")) and d["Debt / Equity"] > 0.75: flags.append("Debt/equity above 0.75")
        if pd.notna(d.get("Debt YoY %")) and d["Debt YoY %"] > 15: flags.append("Debt increased >15% YoY")
        if pd.notna(d.get("Interest Coverage")) and d["Interest Coverage"] < 3: flags.append("Interest coverage below 3x")
        if pd.notna(d.get("ROE %")) and d["ROE %"] < 15: flags.append("ROE below 15%")
        if pd.notna(d.get("ROCE %")) and d["ROCE %"] < 15: flags.append("ROCE below 15%")
        if pd.notna(d.get("Operating Cash Flow")) and d["Operating Cash Flow"] < 0: flags.append("Operating cash flow negative")
        if pd.notna(d.get("Free Cash Flow")) and d["Free Cash Flow"] < 0: flags.append("Free cash flow negative")
        if pd.notna(d.get("OCF / Net Profit %")) and d["OCF / Net Profit %"] < 50: flags.append("Operating cash flow is weak relative to reported profit")
        if pd.notna(d.get("Receivables YoY %")) and pd.notna(d.get("Latest Quarterly Sales YoY %")) and d["Receivables YoY %"] > d["Latest Quarterly Sales YoY %"] + 20: flags.append("Receivables growth materially exceeds recent sales growth")
        if pd.isna(d.get("Promoter Pledge %")): flags.append("Promoter pledge: Yahoo data unavailable")
        d["Red Flags"] = " | ".join(flags) if flags else "No rule-based red flag from available Yahoo fields"

        comps, weighted, available, normalized = score_candidate(d)
        d["Score / 100"] = normalized
        d["Score Coverage %"] = available
        for k,v in comps.items():
            d[f"Score: {k}"] = v * WEIGHTS[k] if pd.notna(v) else np.nan

        # Transparent category assignment; not a recommendation.
        fundamental = 0
        for k in ["3Y Sales CAGR %","3Y Profit CAGR %","ROCE %","ROE %"]:
            if pd.notna(d.get(k)) and d[k] >= (15 if "CAGR" in k else 15): fundamental += 1
        if pd.notna(d.get("Debt / Equity")) and d["Debt / Equity"] < .75: fundamental += 1
        if pd.notna(d.get("Operating Cash Flow")) and d["Operating Cash Flow"] > 0: fundamental += 1
        if pd.notna(d.get("Free Cash Flow")) and d["Free Cash Flow"] > 0: fundamental += 1
        momentum = sum(pd.notna(d.get(k)) and d[k] > 0 for k in ["1M Return %","3M Return %","6M Return %","Price vs 50 DMA %","50 DMA vs 200 DMA %"])
        if fundamental >= 5 and momentum >= 3: d["Screen Category"] = "A — Strong fundamentals + strong momentum"
        elif fundamental >= 5: d["Screen Category"] = "B — Strong fundamentals + emerging momentum"
        elif momentum >= 3: d["Screen Category"] = "C — High momentum but fundamental risks / insufficient evidence"
        else: d["Screen Category"] = "Other / insufficient evidence"

        if not income.empty and not bs.empty and not cf.empty:
            d["Data Status"] = "FULL"
        elif not income.empty or not bs.empty or not cf.empty:
            d["Data Status"] = "PARTIAL"
        else:
            d["Data Status"] = "NO FUNDAMENTALS"
        d["Qualification"] = "PASS" if (
            pd.notna(d.get("Price")) and 20 <= d["Price"] <= 200 and
            pd.notna(d.get("Market Cap ₹Cr")) and d["Market Cap ₹Cr"] >= 500 and
            pd.notna(d.get("Avg Daily Turnover 20D ₹Cr")) and d["Avg Daily Turnover 20D ₹Cr"] >= 0.25
        ) else "FAIL / INCOMPLETE"
    except Exception as exc:
        d["Data Status"] = "ERROR"
        d["Qualification"] = "ERROR"
        d["Error"] = str(exc)
    return d


def apply_screen_filters(df, min_price=20, max_price=200, min_mcap=500, min_turnover_cr=0.25):
    """Apply the basic market/liquidity gate without relying on one exact
    column name. The fast Yahoo prefilter uses `Market Price` and
    `Avg 20D Turnover ₹Cr`, while the full fundamental analysis historically
    used `Price` and `Avg Daily Turnover 20D ₹Cr`. Both are accepted.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()

    price_series = x["Price"] if "Price" in x.columns else x.get("Market Price")
    turnover_series = (
        x["Avg Daily Turnover 20D ₹Cr"]
        if "Avg Daily Turnover 20D ₹Cr" in x.columns
        else x.get("Avg 20D Turnover ₹Cr")
    )
    if price_series is None or turnover_series is None or "Market Cap ₹Cr" not in x.columns:
        x["Qualification Reason"] = "Missing price/turnover/market-cap field"
        return x.iloc[0:0].copy()

    price_num = pd.to_numeric(price_series, errors="coerce")
    turnover_num = pd.to_numeric(turnover_series, errors="coerce")
    mcap_num = pd.to_numeric(x["Market Cap ₹Cr"], errors="coerce")
    mask = (
        price_num.between(min_price, max_price, inclusive="both")
        & (mcap_num >= min_mcap)
        & (turnover_num >= min_turnover_cr)
    )
    out = x.loc[mask].copy()
    out["Screen Price"] = price_num.loc[mask]
    out["Screen Turnover ₹Cr"] = turnover_num.loc[mask]
    return out
