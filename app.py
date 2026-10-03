import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
import re
from urllib.parse import urlparse
import math

from engine import analyze_company_step32
from screener import analyze_candidate, apply_screen_filters, WEIGHTS, fetch_market_universe, prefilter_market_universe, enrich_market_technicals, enrich_market_caps

st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# WATCHLIST — preserved from the uploaded notebook
# ============================================================
DEFAULT_WATCHLIST = {
    "MIC Electronics": "MICEL.NS",
    "ASM Technologies": "ASMTEC.BO",
    "Syrma SGS Technology": "SYRMA.NS",
    "Avalon Technologies": "AVALON.NS",
    "Moschip Technologies": "MOSCHIP.NS",
    "Sahasra Electronic Solutions": "SAHASRA-SM.NS",
    "Motisons Jewellers": "MOTISONS.NS",
    "Sera Investments & Finance": "SERA.BO",
    "Premier Polyfilm": "PREMIERPOL.NS",
    "GACM Technologies": "GATECH.NS",
    "Tembo Global Industries": "TEMBO.NS",
    "UY Fincorp": "UYFINCORP.BO",
    "Bhatia Communications & Retail": "BHATIA.BO",
    "Radhika Jeweltech": "RADHIKAJWE.NS",
    "GTV Engineering": "GTV.BO",
    "Anlon Healthcare": "AHCL.NS",
    "STL Networks": "STLNETWORK.NS",
    "Kaushalya Logistics": "KLL.NS",
    "Quint Digital": "QUINT.NS",
    "Fedders Holding": "FEDDERSHOL.NS",
    "Aaradhya Disposal Industries": "AARADHYA-SM.NS",
    "PVP Ventures": "PVP.NS",
    "Pasupati Acrylon": "PASUPTAC.NS",
    "Sigma Solve": "SIGMA.NS",
    "GenXAI Analytics": "GENXAI-SM.NS",
    "Shukra Pharmaceuticals": "SHUKRAPHAR.NS",
    "Websol Energy System": "WEBELSOLAR.NS",
    "Ardee Industries": "ARDEE.NS",
    "GM Polyplast": "GMPL.BO",
    "Manugraph India": "MANUGRAPH.NS",
    "Vodafone Idea": "IDEA.NS",
    "PCS Technology": "PCS.BO",
    "RKEC Projects": "RKEC.NS",
    "Tuni Textile Mills": "TUNITEX.BO",
    "Grand Foundry": "GFSTEELS.NS",
    "Magnus Steel And Infra": "MAGNUS.BO",
    "Apex Ecotech": "APEXECO-SM.NS",
    "Effwa Infra & Research": "EFFWA.NS",
    "Aeroflex Industries": "AEROFLEX.NS",
    "Venus Pipes & Tubes": "VENUSPIPES.NS",
    "String Metaverse": "META.BO",
    "iStreet Network": "ISTRNETWK.BO",
    "Sarthak Global": "SARTHAKGL.BO",
}

# ============================================================
# Watchlist storage — local JSON or optional Supabase cloud storage
# ============================================================
WATCHLIST_FILE = Path(__file__).resolve().parent / "watchlist.json"
SUPABASE_URL = ""
SUPABASE_KEY = ""

try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

CLOUD_STORAGE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)


def validate_yahoo_symbol(symbol):
    """Validate a Yahoo Finance ticker without requiring the ticker.info endpoint.

    Yahoo/yfinance can return incomplete ``info`` for some Indian small-cap/BSE
    securities even when price history is available. Therefore validation uses
    recent history first and only falls back to metadata.
    """
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return False, "Enter a Yahoo Finance ticker."

    try:
        ticker = yf.Ticker(symbol)

        # Primary check: recent price history. This is generally more reliable
        # than ticker.info for validation and also works for many small caps.
        hist = ticker.history(period="5d", auto_adjust=False, actions=False)
        if hist is not None and not hist.empty:
            close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
            if not close.empty:
                return True, f"Yahoo Finance data found for {symbol}."

        # Secondary check: quote/metadata when history is temporarily absent.
        try:
            fast = ticker.fast_info
            last_price = fast.get("last_price") if hasattr(fast, "get") else None
            if last_price is not None and pd.notna(last_price):
                return True, f"Yahoo Finance quote found for {symbol}."
        except Exception:
            pass

        try:
            info = ticker.get_info()
            if isinstance(info, dict) and (info.get("symbol") or info.get("shortName") or info.get("longName")):
                return True, f"Yahoo Finance security found for {symbol}."
        except Exception:
            pass

        return False, (
            f"Yahoo Finance did not return usable data for {symbol}. "
            "Check the ticker/exchange and try again."
        )
    except Exception as exc:
        return False, f"Could not validate {symbol} with Yahoo Finance: {exc}"


def normalize_symbol(symbol, exchange):
    """Return a Yahoo Finance symbol, adding NSE/BSE suffix when needed."""
    value = str(symbol or "").strip().upper()
    if not value:
        return ""
    if value.endswith(".NS") or value.endswith(".BO"):
        return value
    suffix = ".NS" if exchange == "NSE" else ".BO"
    return value + suffix


def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _cloud_load_watchlist():
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/watchlists"
    params = {"id": "eq.default", "select": "data"}
    r = requests.get(url, headers=_supabase_headers(), params=params, timeout=10)
    r.raise_for_status()
    rows = r.json()
    if rows and isinstance(rows[0].get("data"), dict):
        return {str(k).strip(): str(v).strip().upper() for k, v in rows[0]["data"].items() if str(k).strip() and str(v).strip()}
    return None


def _cloud_save_watchlist(data):
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/watchlists"
    payload = {"id": "default", "data": data}
    headers = _supabase_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()


def save_watchlist(data):
    if CLOUD_STORAGE_ENABLED:
        try:
            _cloud_save_watchlist(data)
            return True, "Cloud watchlist saved."
        except Exception as exc:
            # Keep a local fallback so a temporary cloud error does not lose changes.
            try:
                WATCHLIST_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return False, f"Cloud save failed; local fallback was updated: {exc}"
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, "Local watchlist saved."


def load_watchlist():
    if CLOUD_STORAGE_ENABLED:
        try:
            data = _cloud_load_watchlist()
            if data:
                return data
            data = dict(DEFAULT_WATCHLIST)
            _cloud_save_watchlist(data)
            return data
        except Exception:
            # If cloud storage is temporarily unavailable, fall back to local copy.
            pass
    if not WATCHLIST_FILE.exists():
        data = dict(DEFAULT_WATCHLIST)
        WATCHLIST_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data
    try:
        raw = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = {str(k).strip(): str(v).strip().upper() for k, v in raw.items() if str(k).strip() and str(v).strip()}
            if data:
                return data
    except Exception:
        pass
    data = dict(DEFAULT_WATCHLIST)
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data

watchlist = load_watchlist()

# ============================================================
# Yahoo-like styling
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background:#ffffff; color:#202124; }
    [data-testid="stSidebar"] { background:#f7f7f8; border-right:1px solid #e0e0e0; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
    .yf-header { border-bottom:1px solid #dedede; padding:0.15rem 0 0.8rem 0; margin-bottom:1rem; }
    .yf-brand { font-size:1.65rem; font-weight:800; color:#5f01d1; letter-spacing:-0.5px; }
    .yf-sub { color:#666; font-size:0.86rem; margin-top:0.1rem; }
    .quote-title { font-size:1.8rem; font-weight:750; margin:0; }
    .quote-ticker { color:#666; font-size:0.92rem; margin-top:0.15rem; }
    .quote-price { font-size:2.55rem; font-weight:750; line-height:1.05; margin-top:0.3rem; }
    .quote-meta { color:#666; font-size:0.85rem; }
    .positive { color:#16803c; font-weight:700; }
    .negative { color:#c62828; font-weight:700; }
    .neutral { color:#666; font-weight:600; }
    .section-title { font-size:1.18rem; font-weight:750; border-bottom:1px solid #e4e4e4; padding-bottom:0.45rem; margin-top:1.1rem; }
    .card { border:1px solid #e1e1e1; border-radius:8px; padding:0.8rem 1rem; background:#fff; }
    .small-label { color:#666; font-size:0.78rem; }
    .small-value { font-size:1.05rem; font-weight:650; margin-top:0.15rem; }
    div[data-testid="stMetric"] { border:1px solid #e5e5e5; border-radius:8px; padding:0.55rem 0.75rem; background:#fff; }
    .source-note { color:#777; font-size:0.76rem; }
    .mobile-note { display:none; }

    /* Company Intelligence: prevent mobile cards from truncating with ... */
    .intel-card { border:1px solid #e5e5e5; border-radius:8px; padding:0.65rem 0.75rem; background:#fff; min-height:72px; overflow:visible; }
    .intel-card-label { color:#666; font-size:0.78rem; line-height:1.15; margin-bottom:0.25rem; }
    .intel-card-value { color:#202124; font-size:1.02rem; font-weight:650; line-height:1.25; white-space:normal; overflow-wrap:anywhere; word-break:break-word; }
    .intel-business-summary { white-space:normal; overflow-wrap:anywhere; word-break:break-word; line-height:1.5; }
    .intel-section-note { color:#666; font-size:0.82rem; line-height:1.4; }

    @media (max-width: 768px) {
        .intel-card { min-height:64px; padding:0.55rem 0.65rem; }
        .intel-card-label { font-size:0.72rem; }
        .intel-card-value { font-size:0.92rem; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; max-width:100%; }
        .intel-business-summary { font-size:0.88rem; line-height:1.45; }
        [data-testid="stDataFrame"] { width:100% !important; }
        [data-testid="stDataFrame"] > div { max-width:100% !important; }
        .stAlert p, .stMarkdown p { overflow-wrap:anywhere; word-break:break-word; }
    }
    @media (max-width: 768px) {
        .block-container { padding: 0.65rem 0.65rem 2rem 0.65rem; }
        .yf-brand { font-size:1.25rem; }
        .quote-title { font-size:1.35rem; }
        .quote-price { font-size:2rem; }
        .section-title { font-size:1.05rem; }
        div[data-testid="stMetric"] { padding:0.4rem 0.5rem; }
        .mobile-note { display:block; color:#666; font-size:0.78rem; margin-bottom:0.5rem; }
        [data-testid="stDataFrame"] { font-size:0.78rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Safe formatting helpers
# ============================================================
def num(v):
    try:
        if v is None or pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def fmt_price(v, currency="₹"):
    v = num(v)
    return "—" if pd.isna(v) else f"{currency}{v:,.2f}"


def fmt_num(v, decimals=2):
    v = num(v)
    return "—" if pd.isna(v) else f"{v:,.{decimals}f}"


def fmt_pct(v):
    v = num(v)
    return "—" if pd.isna(v) else f"{v:+.2f}%"


def fmt_crore(v):
    v = num(v)
    return "—" if pd.isna(v) else f"₹{v/1e7:,.2f} Cr"


def first_value(obj, keys, default=np.nan):
    for key in keys:
        try:
            if isinstance(obj, dict) and key in obj:
                return obj[key]
            if hasattr(obj, "get"):
                value = obj.get(key, None)
                if value is not None:
                    return value
        except Exception:
            pass
    return default


def clean_history(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.index = pd.to_datetime(out.index)
    return out


# ============================================================
# Yahoo data layer — each call fails independently
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def quote_data(symbol):
    ticker = yf.Ticker(symbol)
    info = {}
    fast = {}
    try:
        fast = ticker.fast_info
    except Exception:
        fast = {}
    try:
        info = ticker.info
    except Exception:
        info = {}

    hist = pd.DataFrame()
    try:
        hist = clean_history(ticker.history(period="1y", auto_adjust=False))
    except Exception:
        pass

    current = first_value(fast, ["last_price", "lastPrice"])
    previous = first_value(fast, ["previous_close", "previousClose"])
    market_cap = first_value(fast, ["market_cap", "marketCap"])

    if pd.isna(num(current)) and not hist.empty:
        current = hist["Close"].dropna().iloc[-1]
    if pd.isna(num(previous)) and len(hist) >= 2:
        previous = hist["Close"].dropna().iloc[-2]
    if pd.isna(num(market_cap)):
        market_cap = first_value(info, ["marketCap"])

    if not hist.empty:
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        high52 = close.max() if not close.empty else np.nan
        low52 = close.min() if not close.empty else np.nan
    else:
        high52 = first_value(fast, ["year_high", "yearHigh"])
        low52 = first_value(fast, ["year_low", "yearLow"])

    change = np.nan
    if pd.notna(num(current)) and pd.notna(num(previous)) and num(previous) != 0:
        change = (num(current) - num(previous)) / abs(num(previous)) * 100

    def return_from_days(days):
        if hist.empty or "Close" not in hist:
            return np.nan
        s = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(s) < 2:
            return np.nan
        target = s.index[-1] - pd.Timedelta(days=days)
        prior = s.loc[:target]
        if prior.empty or prior.iloc[-1] == 0:
            return np.nan
        return (s.iloc[-1] / prior.iloc[-1] - 1) * 100

    ytd = np.nan
    if not hist.empty and "Close" in hist:
        s = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if not s.empty:
            year_start = pd.Timestamp(s.index[-1].year, 1, 1, tz=s.index[-1].tz) if getattr(s.index[-1], "tzinfo", None) else pd.Timestamp(s.index[-1].year, 1, 1)
            prior = s.loc[:year_start]
            if not prior.empty and prior.iloc[-1] != 0:
                ytd = (s.iloc[-1] / prior.iloc[-1] - 1) * 100

    # IMPORTANT: do not return yfinance FastInfo/Ticker objects from cache_data.
    # Both can contain thread-local/session state that Streamlit cannot pickle.
    # Keep only plain Python scalars in the cached result.
    fast_clean = {}
    for key in [
        "last_price", "previous_close", "market_cap",
        "year_high", "year_low", "year_change",
        "currency", "exchange", "quote_type",
    ]:
        try:
            value = first_value(fast, [key])
            if isinstance(value, (str, int, float, bool)) or value is None:
                fast_clean[key] = value
            elif pd.notna(value):
                fast_clean[key] = float(value)
        except Exception:
            pass

    return {
        "info": info,
        "fast": fast_clean,
        "history": hist,
        "price": num(current),
        "previous": num(previous),
        "change": change,
        "market_cap": num(market_cap),
        "52w_high": num(first_value(info, ["fiftyTwoWeekHigh"], high52)),
        "52w_low": num(first_value(info, ["fiftyTwoWeekLow"], low52)),
        "returns": {
            "1D": change,
            "1W": return_from_days(7),
            "1M": return_from_days(30),
            "3M": return_from_days(90),
            "6M": return_from_days(180),
            "1Y": return_from_days(365),
            "YTD": ytd,
        },
    }


@st.cache_data(ttl=900, show_spinner=False)
def detailed_analysis(company, symbol):
    return analyze_company_step32(company, symbol)


@st.cache_data(ttl=900, show_spinner=False)
def statements(symbol):
    t = yf.Ticker(symbol)
    frames = {}
    for key, attr in [("Annual Income", "income_stmt"), ("Quarterly Income", "quarterly_income_stmt"), ("Annual Balance Sheet", "balance_sheet"), ("Quarterly Balance Sheet", "quarterly_balance_sheet"), ("Annual Cash Flow", "cashflow"), ("Quarterly Cash Flow", "quarterly_cashflow")]:
        try:
            frames[key] = getattr(t, attr)
        except Exception:
            frames[key] = pd.DataFrame()
    return frames


@st.cache_data(ttl=900, show_spinner=False)
def chart_history(symbol, period, interval="1d"):
    try:
        t = yf.Ticker(symbol)
        return clean_history(t.history(period=period, interval=interval, auto_adjust=False))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def company_news(symbol):
    try:
        news = yf.Ticker(symbol).news
        return news if isinstance(news, list) else []
    except Exception:
        return []


# ============================================================
# Company intelligence — evidence-based, Yahoo Finance only
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def company_profile(symbol):
    try:
        info = yf.Ticker(symbol).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _news_datetime(content):
    raw = content.get("pubDate") or content.get("providerPublishTime")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw)
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def normalize_news_items(news):
    rows = []
    for item in news or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        if not isinstance(content, dict):
            continue
        title = content.get("title") or item.get("title") or "Untitled"
        summary = content.get("summary") or content.get("description") or item.get("summary") or ""
        provider = content.get("provider", {})
        publisher = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "Yahoo Finance")
        canonical = content.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else content.get("link") or item.get("link")
        dt = _news_datetime(content)
        text_blob = f"{title} {summary}".lower()
        rows.append({
            "title": str(title).strip(),
            "summary": str(summary).strip(),
            "publisher": publisher or "Yahoo Finance",
            "link": link,
            "datetime": dt,
            "text": text_blob,
        })
    rows.sort(key=lambda x: x["datetime"] or datetime.min, reverse=True)
    return rows


UPCOMING_WORDS = ["upcoming", "expected", "plans to", "planned", "target", "aims to", "by fy", "by q", "commissioning", "will start", "to be completed"]

NEWS_CATEGORIES = {
    "Orders & Contracts": ["order", "contract", "purchase order", "work order", "award", "won", "wins", "deal"],
    "Expansion & Capacity": ["capacity", "expansion", "plant", "facility", "commission", "capex", "manufacturing", "production"],
    "Partnerships & Customers": ["partnership", "partner", "joint venture", "customer", "collaboration", "agreement", "mou"],
    "Products & Technology": ["launch", "product", "technology", "certification", "approval", "patent", "ai", "semiconductor"],
    "Financial Results": ["results", "revenue", "profit", "ebitda", "earnings", "quarter", "q1", "q2", "q3", "q4"],
    "Funding & Capital": ["fundraise", "funding", "debt", "loan", "equity", "preferential", "rights issue", "qip"],
    "Risks / Negative Developments": ["loss", "decline", "delay", "penalty", "regulatory", "investigation", "default", "resign", "downgrade", "fraud"],
}


def classify_news(text):
    matches=[]
    for category, words in NEWS_CATEGORIES.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):
            matches.append(category)
    return matches or ["General Company News"]


def build_company_intelligence(info, result, news_rows):
    positive=[]
    watch=[]
    # Financial evidence from the existing validated Yahoo analysis.
    for key, label in [("Revenue Trend", "Revenue trend"), ("Profit Trend", "Profit trend"), ("EBITDA Trend", "EBITDA trend"),
                       ("Operating CF Trend", "Operating cash flow trend"), ("FCF Trend", "Free cash flow trend"),
                       ("Cash Trend", "Cash trend")]:
        val=result.get(key)
        if val == "INCREASING": positive.append(f"{label}: increasing across the available quarters.")
        elif val == "IMPROVED": positive.append(f"{label}: latest period improved versus the comparison period.")
        elif val == "DECREASING": watch.append(f"{label}: decreasing across the available quarters.")
        elif val == "DECLINED": watch.append(f"{label}: latest period declined versus the comparison period.")
    for key, label in [("Debt Trend", "Debt"), ("Net Debt Trend", "Net debt")]:
        val=result.get(key)
        if val == "DECREASING": positive.append(f"{label}: decreasing across the available quarters.")
        elif val == "INCREASING": watch.append(f"{label}: increasing across the available quarters.")
    margin=result.get("Net Margin Q4 vs Q3 Change")
    if pd.notna(num(margin)):
        (positive if num(margin)>0 else watch if num(margin)<0 else positive).append(
            f"Latest net margin changed {num(margin):+.2f} percentage points QoQ."
        )
    # News evidence. Never convert a headline into a financial fact.
    for n in news_rows[:12]:
        cats=classify_news(n["text"])
        if any(c in cats for c in ["Orders & Contracts","Expansion & Capacity","Partnerships & Customers","Products & Technology"]):
            positive.append(f"Recent company news: {n['title']}")
        if "Risks / Negative Developments" in cats:
            watch.append(f"Recent company news to review: {n['title']}")
    return positive, watch


@st.cache_data(ttl=900, show_spinner=False)
def company_intelligence(symbol, result):
    profile = company_profile(symbol)
    news_rows = normalize_news_items(company_news(symbol))
    positive, watch = build_company_intelligence(profile, result, news_rows)
    return profile, news_rows, positive, watch

# ============================================================
# Sidebar — watchlist + local management
# ============================================================
st.sidebar.markdown("### 📌 Watchlist")
st.sidebar.caption(f"{len(watchlist)} stocks • {'☁️ cloud storage' if CLOUD_STORAGE_ENABLED else '💻 local storage'}")

with st.sidebar.expander("➕ Add stock", expanded=False):
    add_company = st.text_input("Company name", placeholder="Example: Tata Motors", key="add_company")
    add_exchange = st.selectbox("Exchange", ["NSE", "BSE"], key="add_exchange")
    add_symbol_input = st.text_input(
        "Yahoo ticker",
        placeholder="Example: TATAMOTORS or 500570",
        help="Enter the Yahoo Finance ticker. You can enter TATAMOTORS and choose NSE, or 500570 and choose BSE. You may also enter the full ticker such as TATAMOTORS.NS.",
        key="add_symbol",
    )
    full_symbol_preview = normalize_symbol(add_symbol_input, add_exchange)
    if full_symbol_preview:
        st.caption(f"Will use Yahoo ticker: **{full_symbol_preview}**")

    if st.button("Validate & Add to watchlist", type="primary", use_container_width=True):
        clean_company = add_company.strip()
        clean_symbol = normalize_symbol(add_symbol_input, add_exchange)
        if not clean_company or not clean_symbol:
            st.error("Enter both a company name and Yahoo ticker.")
        elif clean_company in watchlist:
            st.error("A company with this name is already in the watchlist.")
        elif clean_symbol in watchlist.values():
            st.error("This Yahoo ticker is already in the watchlist.")
        else:
            with st.spinner(f"Checking {clean_symbol} on Yahoo Finance…"):
                valid, message = validate_yahoo_symbol(clean_symbol)
            if valid:
                watchlist[clean_company] = clean_symbol
                ok, msg = save_watchlist(watchlist)
                if not ok:
                    st.warning(msg)
                st.cache_data.clear()
                st.success(f"Added {clean_company} ({clean_symbol}).")
                st.rerun()
            else:
                st.error(message)
                st.caption("If Yahoo is temporarily unavailable, you can still add the ticker and the app will show missing fields as — until data becomes available.")
                if st.button("Add anyway", use_container_width=True, key="add_anyway"):
                    watchlist[clean_company] = clean_symbol
                    ok, msg = save_watchlist(watchlist)
                    if not ok:
                        st.warning(msg)
                    st.cache_data.clear()
                    st.success(f"Added {clean_company} ({clean_symbol}) to the watchlist.")
                    st.rerun()

company = st.sidebar.selectbox("Select stock", list(watchlist.keys()))
symbol = watchlist[company]
st.sidebar.caption(f"Yahoo Finance ticker: **{symbol}**")

with st.sidebar.expander("🗑️ Remove stock", expanded=False):
    st.caption(f"Selected: **{company} ({symbol})**")
    confirm_remove = st.checkbox("I want to remove this stock", key="confirm_remove")
    if len(watchlist) <= 1:
        st.caption("At least one stock must remain in the watchlist.")
    if st.button("Remove selected stock", use_container_width=True, disabled=(not confirm_remove or len(watchlist) <= 1)):
        del watchlist[company]
        ok, msg = save_watchlist(watchlist)
        if not ok:
            st.warning(msg)
        st.cache_data.clear()
        st.rerun()

with st.sidebar.expander("⚙️ Watchlist settings", expanded=False):
    st.download_button(
        "⬇️ Download watchlist JSON",
        json.dumps(watchlist, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="watchlist.json",
        mime="application/json",
        use_container_width=True,
    )
    if st.button("Reset to built-in 43-stock watchlist", use_container_width=True):
        ok, msg = save_watchlist(DEFAULT_WATCHLIST)
        if not ok:
            st.warning(msg)
        st.cache_data.clear()
        st.rerun()

if st.sidebar.button("🔄 Refresh Yahoo data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("Source: Yahoo Finance via yfinance")
st.sidebar.caption("Watchlist: Supabase cloud storage." if CLOUD_STORAGE_ENABLED else "Watchlist: local watchlist.json. Cloud deployment needs Supabase secrets for persistence.")
st.sidebar.caption("Missing Yahoo fields are shown as —; no values are fabricated.")
if CLOUD_STORAGE_ENABLED:
    st.sidebar.success("☁️ Cloud watchlist persistence is enabled.")


# Cache the lightweight universe scan separately from the expensive
# fundamental analysis. Re-running the same screen in one session should not
# redownload thousands of Yahoo price rows.
@st.cache_data(ttl=1800, show_spinner=False)
def cached_screener_prefilter(universe, min_price, max_price, min_turnover_cr, max_candidates):
    return prefilter_market_universe(
        universe,
        min_price=min_price,
        max_price=max_price,
        min_turnover_cr=min_turnover_cr,
        max_candidates=max_candidates,
    )

@st.cache_data(ttl=1800, show_spinner=False)
def cached_screener_market_caps(candidates):
    return enrich_market_caps(candidates, max_workers=12)


# ============================================================
# Sidebar — automatic NSE/BSE growth screener
# ============================================================
st.sidebar.divider()
with st.sidebar.expander("🔎 Growth Screener", expanded=False):
    st.caption("Automatically discovers the NSE/BSE equity universe and screens it with Yahoo Finance data. No CSV upload is required.")
    screener_min_price = st.number_input("Min price ₹", 1.0, 1000.0, 20.0, 1.0, key="scr_min_price")
    screener_max_price = st.number_input("Max price ₹", 20.0, 5000.0, 200.0, 1.0, key="scr_max_price")
    screener_min_mcap = st.number_input("Min market cap ₹Cr", 0.0, 1000000.0, 500.0, 50.0, key="scr_min_mcap")
    screener_min_turnover = st.number_input("Min avg daily turnover ₹Cr", 0.0, 10000.0, 0.25, 0.25, key="scr_min_turnover")
    screener_strict = st.checkbox("Strict fundamental gate", value=False, key="scr_strict", help="Apply the requested growth, leverage, profitability and positive cash-flow thresholds before showing qualified companies.")
    screener_include_sme = st.checkbox("Include NSE SME", value=False, key="scr_include_sme_v13", help="Off by default because the objective excludes SME/illiquid characteristics.")
    screener_include_bse = st.checkbox("Include BSE", value=True, key="scr_include_bse_v13")
    screener_max_scan = st.number_input("Maximum liquid candidates", min_value=50, max_value=1000, value=300, step=50, key="scr_max_scan", help="After the fast price/liquidity scan, market-cap data is checked for these candidates before expensive fundamental analysis.")
    screener_max_fundamentals = st.number_input("Maximum fundamental analyses", min_value=10, max_value=200, value=50, step=10, key="scr_max_fundamentals", help="Only market-cap-qualified candidates up to this number receive the slower financial-statement analysis.")

    if st.button("🌐 Fetch NSE/BSE universe", use_container_width=True, key="fetch_universe"):
        with st.spinner("Fetching the latest exchange security lists…"):
            uni, uni_errors = fetch_market_universe(include_sme=screener_include_sme, include_bse=screener_include_bse)
        st.session_state["screener_universe"] = uni
        st.session_state["screener_universe_errors"] = uni_errors
        st.session_state["screener_universe_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    universe = st.session_state.get("screener_universe")
    if isinstance(universe, pd.DataFrame) and not universe.empty:
        st.caption(f"Universe: {len(universe):,} securities")
        if st.button("🚀 Fetch + run automatic screen", type="primary", use_container_width=True, key="run_auto_screen"):
            # The market prefilter is now cached for 30 minutes and uses larger
            # Yahoo batches, so repeated clicks do not restart a long scan.
            with st.spinner("Scanning Yahoo prices/volume in batches…"):
                pre = cached_screener_prefilter(
                    universe,
                    float(screener_min_price),
                    float(screener_max_price),
                    float(screener_min_turnover),
                    int(screener_max_scan),
                )
            if pre.empty:
                st.warning("No securities passed the initial price/liquidity filter. Try a lower liquidity threshold or wider price range.")
            else:
                # IMPORTANT: market cap is checked before expensive fundamentals.
                # Yahoo fast_info is not consistently populated for Indian
                # listings, so enrich_market_caps falls back to Yahoo shares.
                with st.spinner(f"Checking market caps for {len(pre):,} liquid candidates…"):
                    pre = cached_screener_market_caps(pre)
                pre_market = apply_screen_filters(pre, screener_min_price, screener_max_price, screener_min_mcap, screener_min_turnover)
                cap_available = pd.to_numeric(pre.get("Market Cap ₹Cr", pd.Series(np.nan, index=pre.index)), errors="coerce").notna().sum()
                st.caption(
                    f"Fast filter: {len(pre):,} liquid candidates | market-cap data available: {cap_available:,}/{len(pre):,} | "
                    f"market-cap qualified: {len(pre_market):,} at ₹{screener_min_mcap:,.0f}Cr minimum. "
                    f"Fundamental analysis is limited to the top {int(screener_max_fundamentals):,} market-qualified candidates."
                )
                # One batched 1-year request only for the market-qualified
                # candidates. This prevents technical/fundamental work on
                # stocks that fail the user's basic market-cap constraint.
                pre = pre_market.sort_values("Avg 20D Turnover ₹Cr", ascending=False).head(int(screener_max_fundamentals)).reset_index(drop=True)
                with st.spinner(f"Loading technicals for {len(pre):,} market-qualified candidates…"):
                    pre = enrich_market_technicals(pre)
                st.session_state["screener_prefilter"] = pre
                if pre.empty:
                    st.warning(
                        f"No company passed the basic market gate at the current settings. "
                        f"Remember: 'Min market cap ₹Cr' is a minimum; a ₹1,842 Cr company does not qualify when the minimum is ₹5,000 Cr."
                    )
                rows = []
                progress = st.progress(0)
                status = st.empty()

                # Fundamental analysis is much more expensive than the price
                # scan because each ticker may require info + income + balance
                # sheet + cash flow + quarterly statements. Use a small worker
                # pool instead of processing candidates strictly one-by-one.
                def _analyze_one(rr):
                    try:
                        return analyze_candidate(
                            str(rr.Company), str(rr.Symbol), str(rr.Exchange),
                            market_price=getattr(rr, "Market Price", np.nan),
                            market_turnover=getattr(rr, "Avg 20D Turnover ₹Cr", np.nan),
                        )
                    except Exception as exc:
                        return {
                            "Company": str(rr.Company),
                            "Symbol": str(rr.Symbol),
                            "Exchange": str(rr.Exchange),
                            "Scan Error": str(exc),
                        }

                workers = min(8, max(2, len(pre)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_analyze_one, rr) for rr in pre.itertuples(index=False)]
                    total = len(futures)
                    for i, future in enumerate(as_completed(futures), 1):
                        rows.append(future.result())
                        status.write(f"Analyzing fundamentals: {i}/{total} completed…")
                        progress.progress(int(i / total * 100))

                results_df = pd.DataFrame(rows)
                # The fast batch scan already calculated price, turnover and
                # technical momentum. Merge those fields back so the expensive
                # fundamental worker does not need another Yahoo history call.
                market_cols = [c for c in pre.columns if c not in {"Company", "Exchange"}]
                if "Symbol" in pre.columns and "Symbol" in results_df.columns:
                    market_snapshot = pre[market_cols].drop_duplicates("Symbol")
                    results_df = results_df.drop(columns=[c for c in market_snapshot.columns if c != "Symbol" and c in results_df.columns], errors="ignore")
                    results_df = results_df.merge(market_snapshot, on="Symbol", how="left", suffixes=("", "_market"))
                    for c in [x for x in results_df.columns if x.endswith("_market")]:
                        base = c[:-7]
                        if base in results_df.columns:
                            results_df[base] = results_df[base].where(results_df[base].notna(), results_df[c])
                            results_df.drop(columns=[c], inplace=True)
                        else:
                            results_df.rename(columns={c: base}, inplace=True)
                st.session_state["screener_results"] = results_df
                status.success(f"Screen complete: {len(rows):,} securities analyzed using {workers} parallel workers.")
                progress.empty()
    else:
        st.info("Click 'Fetch NSE/BSE universe' to discover securities automatically.")

    if st.session_state.get("screener_universe_errors"):
        st.caption("Some exchange feeds were unavailable: " + " | ".join(st.session_state["screener_universe_errors"][:2]))
    if st.session_state.get("screener_universe_date"):
        st.caption("Universe fetched: " + st.session_state["screener_universe_date"])

# ============================================================
# Load quote
# ============================================================
with st.spinner(f"Loading {company}…"):
    q = quote_data(symbol)
    result = detailed_analysis(company, symbol)

info = q["info"]
hist = q["history"]
price = q["price"]
prev = q["previous"]
change = q["change"]

name = first_value(info, ["longName", "shortName"], company)
sector = first_value(info, ["sector"], "")
industry = first_value(info, ["industry"], "")
exchange = first_value(info, ["fullExchangeName", "exchange"], symbol.split(".")[-1])
currency = first_value(info, ["currency"], "INR")
currency_symbol = "₹" if currency == "INR" else f"{currency} "

# ============================================================
# Header
# ============================================================
st.markdown(
    f"<div class='yf-header'><span class='yf-brand'>Yahoo-style Stock Research</span><div class='yf-sub'>Read-only market, technical and financial research dashboard</div></div>",
    unsafe_allow_html=True,
)

h1, h2 = st.columns([2.7, 1.3])
with h1:
    st.markdown(f"<div class='quote-title'>{name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='quote-ticker'>{symbol} · {exchange} · {sector or 'Sector unavailable'}{(' · ' + industry) if industry else ''}</div>", unsafe_allow_html=True)
    price_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
    st.markdown(f"<div class='quote-price'>{fmt_price(price, currency_symbol)}</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='{price_class}'>{fmt_pct(change)}</span> today <span class='quote-meta'>· Previous close {fmt_price(prev, currency_symbol)}</span>", unsafe_allow_html=True)
with h2:
    st.metric("Market Cap", fmt_crore(q["market_cap"]))
    st.metric("52W Range", f"{fmt_price(q['52w_low'], currency_symbol)} – {fmt_price(q['52w_high'], currency_symbol)}")

st.markdown("<div class='source-note'>Yahoo Finance data may be delayed and availability varies by ticker/exchange.</div>", unsafe_allow_html=True)

# ============================================================
# Automatic growth screener results (controls live in the sidebar)
# ============================================================
results = st.session_state.get("screener_results")
if isinstance(results, pd.DataFrame) and not results.empty:
    st.markdown("## 🔎 NSE + BSE Growth Screener")
    st.caption("Screen controls are in the left sidebar. Exchange lists are fetched automatically; Yahoo Finance supplies the market and financial data. Missing fields remain —.")
    # Diagnose each market gate separately so an empty result is actionable
    # instead of appearing as a generic blank table.
    _price_col = results["Price"] if "Price" in results.columns else results.get("Market Price", pd.Series(np.nan, index=results.index))
    _turn_col = results["Avg Daily Turnover 20D ₹Cr"] if "Avg Daily Turnover 20D ₹Cr" in results.columns else results.get("Avg 20D Turnover ₹Cr", pd.Series(np.nan, index=results.index))
    _mcap_col = results.get("Market Cap ₹Cr", pd.Series(np.nan, index=results.index))
    _price_n = pd.to_numeric(_price_col, errors="coerce")
    _turn_n = pd.to_numeric(_turn_col, errors="coerce")
    _mcap_n = pd.to_numeric(_mcap_col, errors="coerce")
    _price_pass = _price_n.between(screener_min_price, screener_max_price, inclusive="both")
    _turn_pass = _turn_n >= screener_min_turnover
    _mcap_pass = _mcap_n >= screener_min_mcap
    _market_cap_available = _mcap_n.notna()
    st.caption(
        f"Market gate diagnostics — price: {_price_pass.sum():,}/{len(results):,} | "
        f"turnover: {_turn_pass.sum():,}/{len(results):,} | "
        f"market cap ≥ ₹{screener_min_mcap:,.0f}Cr: {_mcap_pass.sum():,}/{len(results):,} | "
        f"market-cap data available: {_market_cap_available.sum():,}/{len(results):,}"
    )
    qualified = apply_screen_filters(results, screener_min_price, screener_max_price, screener_min_mcap, screener_min_turnover)
    base_qualified_count = len(qualified)
    if screener_strict and not qualified.empty:
        numeric = lambda c: pd.to_numeric(qualified[c], errors="coerce") if c in qualified.columns else pd.Series(np.nan, index=qualified.index)
        criteria = {
            "3Y Sales CAGR %": lambda s: s > 15,
            "3Y Profit CAGR %": lambda s: s > 20,
            "Latest Quarterly Sales YoY %": lambda s: s > 10,
            "Latest Quarterly Profit YoY %": lambda s: s > 20,
            "Debt / Equity": lambda s: s < 0.75,
            "ROCE %": lambda s: s > 15,
            "ROE %": lambda s: s > 15,
            "Operating Margin %": lambda s: s > 10,
            "Net Margin %": lambda s: s > 8,
            "Operating Cash Flow": lambda s: s > 0,
            "Free Cash Flow": lambda s: s > 0,
        }
        passed_count = pd.Series(0, index=qualified.index, dtype="int64")
        available_count = pd.Series(0, index=qualified.index, dtype="int64")
        for col, rule in criteria.items():
            s = numeric(col)
            available = s.notna()
            passed_count += rule(s.fillna(-np.inf)).astype(int)
            available_count += available.astype(int)
        # Require every available criterion to pass, but do not let a Yahoo
        # missing field turn a real company into an automatic failure. A 70%
        # coverage floor keeps the gate meaningful while exposing incomplete
        # data explicitly.
        strict_mask = (available_count >= math.ceil(len(criteria) * 0.70)) & (passed_count == available_count)
        qualified = qualified.loc[strict_mask].copy()
        if qualified.empty and base_qualified_count:
            st.warning(
                f"Strict gate returned 0 of {base_qualified_count:,} market-qualified stocks. "
                "The gate now requires at least 70% of the fundamental fields to be available and all available fields to pass. "
                "Check the Data Status / Red Flags columns or turn off Strict fundamental gate to inspect the full research set."
            )
    st.caption(f"Market/liquidity qualified before fundamental gate: **{base_qualified_count:,}** | Final qualified: **{len(qualified):,}** | Securities analyzed: **{len(results):,}**")
    if base_qualified_count == 0:
        st.warning(
            "No company passed the basic market gate. If market-cap data available is 0, "
            "this was a data-coverage problem rather than evidence that the universe contains no qualifying companies. "
            "The screener now derives market cap from price × shares outstanding when Yahoo fast_info does not provide it."
        )
    cat = st.multiselect("Category", ["A — Strong fundamentals + strong momentum","B — Strong fundamentals + emerging momentum","C — High momentum but fundamental risks / insufficient evidence"], default=[], key="screen_category_filter")
    view = qualified.copy()
    if cat:
        view = view[view["Screen Category"].isin(cat)]
    if "Score / 100" in view:
        view = view.sort_values("Score / 100", ascending=False, na_position="last")
    display_cols=["Company","Exchange","Symbol","Price","Market Cap ₹Cr","Market Cap Source","Data Status","1M Return %","3M Return %","3Y Sales CAGR %","3Y Profit CAGR %","Latest Quarterly Sales YoY %","Latest Quarterly Profit YoY %","ROCE %","ROE %","Debt / Equity","Debt Trend","Free Cash Flow","Promoter Holding %","Promoter Pledge %","P/E","Industry P/E","Screen Category","Score / 100","Score Coverage %","Red Flags"]
    display_cols=[c for c in display_cols if c in view.columns]
    st.dataframe(view[display_cols].head(25), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download screened results CSV", view.to_csv(index=False).encode("utf-8"), file_name="yahoo_growth_screener_results.csv", mime="text/csv", key="download_screen_results")
    st.markdown("### Score breakdown")
    breakdown_cols=["Company","Score / 100","Score Coverage %"] + [f"Score: {k}" for k in WEIGHTS]
    breakdown_cols=[c for c in breakdown_cols if c in view.columns]
    st.dataframe(view[breakdown_cols].head(25), use_container_width=True, hide_index=True)
    st.caption("The 100-point framework is transparent. Missing components are excluded from the denominator and reported through Score Coverage %. This is a research screen, not a BUY/SELL recommendation.")
    for title, category in [("A — Strong Fundamentals + Strong Momentum","A — Strong fundamentals + strong momentum"),("B — Strong Fundamentals + Emerging Momentum","B — Strong fundamentals + emerging momentum"),("C — High Momentum but Fundamental Risks","C — High momentum but fundamental risks / insufficient evidence")]:
        st.markdown(f"### {title}")
        block=view[view["Screen Category"]==category].head(25)
        cols=[c for c in ["Company","Symbol","Price","3Y Sales CAGR %","3Y Profit CAGR %","ROCE %","ROE %","Debt / Equity","Free Cash Flow","1M Return %","3M Return %","6M Return %","Score / 100","Red Flags"] if c in block.columns]
        st.dataframe(block[cols], use_container_width=True, hide_index=True)
    st.markdown("### 5 deepest-research candidates")
    deep=view[view["Screen Category"].isin(["A — Strong fundamentals + strong momentum","B — Strong fundamentals + emerging momentum"])].copy()
    deep=deep.sort_values(["Score / 100","Score Coverage %"], ascending=[False,False], na_position="last").head(5)
    st.dataframe(deep[[c for c in ["Company","Symbol","Price","Market Cap ₹Cr","3Y Sales CAGR %","3Y Profit CAGR %","ROCE %","ROE %","Debt / Equity","Debt Trend","Operating Cash Flow","Free Cash Flow","P/E","1M Return %","3M Return %","6M Return %","Screen Category","Score / 100","Score Coverage %","Red Flags"] if c in deep.columns]], use_container_width=True, hide_index=True)
    st.divider()

# ============================================================
# Navigation tabs — Yahoo-like quote page structure
# ============================================================
tab_summary, tab_intel, tab_chart, tab_statistics, tab_financials, tab_quarterly, tab_news, tab_raw = st.tabs([
    "Summary", "Company Intelligence", "Chart", "Statistics", "Financials", "Quarterly", "News", "Raw Data"
])

# ============================================================
# COMPANY INTELLIGENCE
# ============================================================
with tab_intel:
    st.markdown("<div class='section-title'>Company Intelligence</div>", unsafe_allow_html=True)
    st.caption(
        "Dynamic company research for the currently selected Yahoo Finance ticker. "
        "The same intelligence view is generated for every stock in the watchlist, including stocks added later."
    )

    # The selected ticker drives everything here; no hard-coded company universe is used.
    try:
        intel_profile, intel_news, intel_positive, intel_watch = company_intelligence(symbol, result)
    except Exception as exc:
        intel_profile = info if isinstance(info, dict) else {}
        intel_news = normalize_news_items(company_news(symbol))
        intel_positive, intel_watch = build_company_intelligence(
            intel_profile, result if isinstance(result, dict) else {}, intel_news
        )
        st.warning(f"Some Company Intelligence fields could not be loaded: {exc}")

    # -------- Company identity / business description --------
    intel_name = first_value(
        intel_profile,
        ["longName", "shortName", "displayName"],
        name or company
    )
    intel_sector = first_value(intel_profile, ["sector"], sector or "—")
    intel_industry = first_value(intel_profile, ["industry"], industry or "—")
    intel_country = first_value(intel_profile, ["country"], "—")
    intel_exchange = first_value(
        intel_profile,
        ["fullExchangeName", "exchange"],
        exchange or symbol.split(".")[-1]
    )
    intel_currency = first_value(intel_profile, ["currency"], currency or "INR")
    intel_employees = first_value(intel_profile, ["fullTimeEmployees"], np.nan)
    intel_website = first_value(intel_profile, ["website"], "")
    intel_summary = str(first_value(
        intel_profile,
        ["longBusinessSummary", "businessSummary"],
        ""
    ) or "").strip()

    ic1, ic2 = st.columns(2)
    ic3, ic4 = st.columns(2)

    def intel_card(label, value):
        safe_label = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_value = str(value if value not in (None, "") else "—")
        safe_value = safe_value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<div class='intel-card'><div class='intel-card-label'>{safe_label}</div><div class='intel-card-value'>{safe_value}</div></div>"

    with ic1: st.markdown(intel_card("Sector", intel_sector), unsafe_allow_html=True)
    with ic2: st.markdown(intel_card("Industry", intel_industry), unsafe_allow_html=True)
    with ic3: st.markdown(intel_card("Country", intel_country), unsafe_allow_html=True)
    with ic4: st.markdown(intel_card("Employees", fmt_num(intel_employees, 0)), unsafe_allow_html=True)

    if intel_summary:
        with st.expander("🏢 What the company does", expanded=True):
            st.markdown(f"<div class='intel-business-summary'>{intel_summary}</div>", unsafe_allow_html=True)
            if intel_website:
                st.markdown(f"[Company website]({intel_website})")
    else:
        st.info(
            "Yahoo Finance did not provide a business description for this ticker. "
            "The financial, market and news sections below remain available."
        )

    # -------- Key research snapshot --------
    st.markdown("<div class='section-title'>Research snapshot</div>", unsafe_allow_html=True)

    def intel_value(value, formatter=None):
        try:
            if formatter:
                return formatter(value)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "—"
            return str(value)
        except Exception:
            return "—"

    roe = first_value(intel_profile, ["returnOnEquity"])
    roa = first_value(intel_profile, ["returnOnAssets"])
    debt_to_equity = first_value(intel_profile, ["debtToEquity"])
    profit_margin = first_value(intel_profile, ["profitMargins"])
    operating_margin = first_value(intel_profile, ["operatingMargins"])
    revenue_growth = first_value(intel_profile, ["revenueGrowth"])
    earnings_growth = first_value(intel_profile, ["earningsGrowth"])
    fcf = first_value(intel_profile, ["freeCashflow"])
    ocf = first_value(intel_profile, ["operatingCashflow"])
    total_debt = first_value(intel_profile, ["totalDebt"])
    cash = first_value(intel_profile, ["totalCash"])

    def pct_or_dash(v):
        n = num(v)
        if pd.isna(n):
            return "—"
        # Yahoo commonly returns margins/growth as decimals.
        if abs(n) <= 2:
            n *= 100
        return f"{n:+.2f}%"

    def crore_from_value(v):
        n = num(v)
        return "—" if pd.isna(n) else f"₹{n/1e7:,.2f} Cr"

    snapshot_rows = [
        ["Market Cap", fmt_crore(q.get("market_cap"))],
        ["P/E", fmt_num(first_value(intel_profile, ["trailingPE"]))],
        ["Forward P/E", fmt_num(first_value(intel_profile, ["forwardPE"]))],
        ["P/B", fmt_num(first_value(intel_profile, ["priceToBook"]))],
        ["ROE", pct_or_dash(roe)],
        ["ROA", pct_or_dash(roa)],
        ["Debt / Equity", fmt_num(debt_to_equity)],
        ["Profit Margin", pct_or_dash(profit_margin)],
        ["Operating Margin", pct_or_dash(operating_margin)],
        ["Revenue Growth", pct_or_dash(revenue_growth)],
        ["Earnings Growth", pct_or_dash(earnings_growth)],
        ["Operating Cash Flow", crore_from_value(ocf)],
        ["Free Cash Flow", crore_from_value(fcf)],
        ["Total Debt", crore_from_value(total_debt)],
        ["Cash", crore_from_value(cash)],
    ]
    snapshot_df = pd.DataFrame(snapshot_rows, columns=["Metric", "Value"])
    for start in range(0, len(snapshot_df), 2):
        row_cols = st.columns(2)
        for j, col in enumerate(row_cols):
            idx = start + j
            if idx >= len(snapshot_df):
                continue
            with col:
                st.markdown(intel_card(snapshot_df.iloc[idx]["Metric"], snapshot_df.iloc[idx]["Value"]), unsafe_allow_html=True)

    # -------- Evidence from the existing analyzer --------
    st.markdown("<div class='section-title'>Fundamental & market evidence</div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Positive / improving evidence**")
        positive_items = list(intel_positive or [])

        # Also include the analyzer's own evidence fields, which are available
        # for every ticker that successfully passes through detailed_analysis.
        analyzer_positive = result.get("Positive Evidence", "") if isinstance(result, dict) else ""
        if analyzer_positive:
            positive_items.extend(
                x.strip() for x in str(analyzer_positive).split("|") if x.strip()
            )

        # Deduplicate while preserving order.
        seen = set()
        positive_items = [
            x for x in positive_items
            if not (x in seen or seen.add(x))
        ]

        if positive_items:
            for item in positive_items[:15]:
                st.success("✓ " + item)
        else:
            st.info("No clear positive evidence was returned by the available Yahoo/analyzer fields.")

    with e2:
        st.markdown("**Risks / items to review**")
        watch_items = list(intel_watch or [])
        analyzer_negative = result.get("Negative / Watch Evidence", "") if isinstance(result, dict) else ""
        if analyzer_negative:
            watch_items.extend(
                x.strip() for x in str(analyzer_negative).split("|") if x.strip()
            )

        seen = set()
        watch_items = [
            x for x in watch_items
            if not (x in seen or seen.add(x))
        ]

        if watch_items:
            for item in watch_items[:15]:
                st.warning("• " + item)
        else:
            st.info("No clear risk/watch item was returned by the available Yahoo/analyzer fields.")

    # -------- Business/news catalysts --------
    st.markdown("<div class='section-title'>Company-specific catalysts & recent developments</div>", unsafe_allow_html=True)

    if intel_news:
        category_filter = st.multiselect(
            "Filter company developments",
            sorted({c for n in intel_news for c in classify_news(n["text"])}),
            default=[],
            key=f"intel_news_filter_{symbol}",
        )

        shown = 0
        for n in intel_news:
            cats = classify_news(n["text"])
            if category_filter and not any(c in category_filter for c in cats):
                continue

            date_text = (
                n["datetime"].strftime("%d %b %Y %H:%M")
                if n["datetime"] else "Date unavailable"
            )

            st.markdown(f"**{n['title']}**")
            st.caption(
                f"{n['publisher']} · {date_text} · {', '.join(cats)}"
            )
            if n["summary"]:
                st.write(n["summary"][:700])
            if n["link"]:
                st.markdown(f"[Open source article]({n['link']})")

            shown += 1
            if shown >= 15:
                break

        if shown == 0:
            st.info("No company developments matched the selected topics.")
    else:
        st.info(
            "Yahoo Finance did not return company news for this ticker. "
            "This does not mean the company has no news; it means the current public Yahoo feed had no usable items."
        )

    # -------- Data coverage / transparency --------
    st.markdown("<div class='section-title'>Company Intelligence data coverage</div>", unsafe_allow_html=True)

    coverage_fields = {
        "Business description": bool(intel_summary),
        "Sector": bool(intel_sector and intel_sector != "—"),
        "Industry": bool(intel_industry and intel_industry != "—"),
        "Employees": pd.notna(num(intel_employees)),
        "Valuation data": any(pd.notna(num(first_value(intel_profile, [k]))) for k in ["trailingPE", "forwardPE", "priceToBook"]),
        "Profitability data": any(pd.notna(num(first_value(intel_profile, [k]))) for k in ["returnOnEquity", "returnOnAssets", "profitMargins"]),
        "Balance-sheet data": any(pd.notna(num(first_value(intel_profile, [k]))) for k in ["totalDebt", "totalCash", "debtToEquity"]),
        "Cash-flow data": any(pd.notna(num(first_value(intel_profile, [k]))) for k in ["operatingCashflow", "freeCashflow"]),
        "Company news": bool(intel_news),
        "Analyzer financial trends": bool(result),
    }

    coverage_df = pd.DataFrame(
        [[k, "Available" if v else "Not available"] for k, v in coverage_fields.items()],
        columns=["Intelligence component", "Status"]
    )
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    st.caption(
        f"Ticker: {symbol} · Exchange: {intel_exchange} · Currency: {intel_currency}. "
        "Data comes from Yahoo Finance/yfinance and the existing analyzer. "
        "Missing fields are shown as unavailable rather than fabricated. "
        "The same code path is used for every selected or newly added stock."
    )

# ============================================================
# SUMMARY
# ============================================================
with tab_summary:
    st.markdown("<div class='section-title'>Key quote statistics</div>", unsafe_allow_html=True)
    k = st.columns(7)
    summary_kpis = [
        ("Market Cap", fmt_crore(q["market_cap"])),
        ("P/E", fmt_num(first_value(info, ["trailingPE"]))),
        ("Forward P/E", fmt_num(first_value(info, ["forwardPE"]))),
        ("P/B", fmt_num(first_value(info, ["priceToBook"]))),
        ("EV/EBITDA", fmt_num(first_value(info, ["enterpriseToEbitda"]))),
        ("Dividend Yield", fmt_pct((num(first_value(info, ["dividendYield"])) * 100) if pd.notna(num(first_value(info, ["dividendYield"]))) and num(first_value(info, ["dividendYield"])) < 1 else first_value(info, ["dividendYield"]))),
        ("Beta", fmt_num(first_value(info, ["beta"]))),
    ]
    for col, (label, value) in zip(k, summary_kpis):
        col.metric(label, value)

    st.markdown("<div class='section-title'>Performance</div>", unsafe_allow_html=True)
    perf_cols = st.columns(7)
    for col, (label, value) in zip(perf_cols, q["returns"].items()):
        col.metric(label, fmt_pct(value))

    st.markdown("<div class='section-title'>52-week position</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("52W High", fmt_price(q["52w_high"], currency_symbol))
    c2.metric("52W Low", fmt_price(q["52w_low"], currency_symbol))
    high_distance = ((price - q["52w_high"]) / q["52w_high"] * 100) if pd.notna(price) and pd.notna(q["52w_high"]) and q["52w_high"] else np.nan
    low_distance = ((price - q["52w_low"]) / q["52w_low"] * 100) if pd.notna(price) and pd.notna(q["52w_low"]) and q["52w_low"] else np.nan
    c3.metric("Distance from 52W High", fmt_pct(high_distance))

    st.markdown("<div class='section-title'>Latest quarterly snapshot — easy comparison</div>", unsafe_allow_html=True)
    if result.get("Latest Quarter Date"):
        latest_date = result.get("Latest Quarter Date")
        previous_date = result.get("Previous Quarter Date")
        st.caption(f"Latest quarter: **{latest_date}** · Previous quarter: **{previous_date or '—'}** · Money values are ₹ crore")

    def direction(change):
        change = num(change)
        if pd.isna(change):
            return "—"
        if change > 0.005:
            return "↑ Increased"
        if change < -0.005:
            return "↓ Decreased"
        return "→ No major change"

    def quarter_money(current, previous, qoq):
        return [fmt_crore(current), fmt_crore(previous), fmt_pct(qoq), direction(qoq)]

    def quarter_margin(current, previous, change):
        return [
            ("—" if pd.isna(num(current)) else f"{num(current):.2f}%"),
            ("—" if pd.isna(num(previous)) else f"{num(previous):.2f}%"),
            ("—" if pd.isna(num(change)) else f"{num(change):+.2f} pp"),
            direction(change),
        ]

    rows = []
    for label, cur_key, prev_key, change_key in [
        ("Revenue", "Revenue Q4", "Revenue Q3", "Revenue Q4 vs Q3 %"),
        ("Net Profit", "Profit Q4", "Profit Q3", "Profit Q4 vs Q3 %"),
        ("EBITDA", "EBITDA Q4", "EBITDA Q3", "EBITDA Q4 vs Q3 %"),
        ("Debt", "Debt Q4", "Debt Q3", "Debt Q4 vs Q3 %"),
        ("Cash", "Cash Q4", "Cash Q3", "Cash Q4 vs Q3 %"),
        ("Net Debt", "Net Debt Q4", "Net Debt Q3", "Net Debt Q4 vs Q3 %"),
        ("Operating Cash Flow", "Operating CF Q4", "Operating CF Q3", "Operating CF Q4 vs Q3 %"),
        ("Free Cash Flow", "FCF Q4", "FCF Q3", "FCF Q4 vs Q3 %"),
    ]:
        vals = quarter_money(result.get(cur_key), result.get(prev_key), result.get(change_key))
        rows.append([label, *vals])

    margin_vals = quarter_margin(result.get("Net Margin Q4 %"), result.get("Net Margin Q3 %"), result.get("Net Margin Q4 vs Q3 Change"))
    rows.insert(3, ["Net Margin", *margin_vals])

    q_snapshot = pd.DataFrame(rows, columns=["Metric", "Latest Quarter", "Previous Quarter", "QoQ Change", "Direction"])
    st.dataframe(q_snapshot, use_container_width=True, hide_index=True, column_config={
        "Metric": st.column_config.TextColumn("Metric", width="medium"),
        "Latest Quarter": st.column_config.TextColumn("Latest Quarter", width="medium"),
        "Previous Quarter": st.column_config.TextColumn("Previous Quarter", width="medium"),
        "QoQ Change": st.column_config.TextColumn("QoQ Change", width="small"),
        "Direction": st.column_config.TextColumn("Direction", width="medium"),
    })
    st.info("How to read it: ↑ Increased means the latest quarter is higher than the previous quarter. ↓ Decreased means it is lower. For Net Margin, QoQ Change is shown in percentage points (pp).")

    st.markdown("<div class='section-title'>Quarterly trend charts</div>", unsafe_allow_html=True)
    chart_dates = [result.get("Quarter -4 Date"), result.get("Quarter -3 Date"), result.get("Previous Quarter Date"), result.get("Latest Quarter Date")]
    chart_dates = [d if d else f"Q{i+1}" for i,d in enumerate(chart_dates)]
    revenue_vals = [num(result.get(k))/1e7 if pd.notna(num(result.get(k))) else np.nan for k in ["Revenue Q1","Revenue Q2","Revenue Q3","Revenue Q4"]]
    profit_vals = [num(result.get(k))/1e7 if pd.notna(num(result.get(k))) else np.nan for k in ["Profit Q1","Profit Q2","Profit Q3","Profit Q4"]]
    ebitda_vals = [num(result.get(k))/1e7 if pd.notna(num(result.get(k))) else np.nan for k in ["EBITDA Q1","EBITDA Q2","EBITDA Q3","EBITDA Q4"]]
    fig_f = go.Figure()
    for vals, label in [(revenue_vals,"Revenue"),(profit_vals,"Net Profit"),(ebitda_vals,"EBITDA")]:
        fig_f.add_trace(go.Scatter(x=chart_dates, y=vals, mode="lines+markers", name=label))
    fig_f.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified", yaxis_title="₹ crore")
    st.plotly_chart(fig_f, use_container_width=True)

    debt_vals = [num(result.get(k))/1e7 if pd.notna(num(result.get(k))) else np.nan for k in ["Debt Q1","Debt Q2","Debt Q3","Debt Q4"]]
    net_debt_vals = [num(result.get(k))/1e7 if pd.notna(num(result.get(k))) else np.nan for k in ["Net Debt Q1","Net Debt Q2","Net Debt Q3","Net Debt Q4"]]
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=chart_dates, y=debt_vals, mode="lines+markers", name="Debt"))
    fig_d.add_trace(go.Scatter(x=chart_dates, y=net_debt_vals, mode="lines+markers", name="Net Debt"))
    fig_d.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified", yaxis_title="₹ crore")
    st.plotly_chart(fig_d, use_container_width=True)
    # ========================================================
    # EASY-TO-UNDERSTAND MARKET + FUNDAMENTAL ANALYSIS
    # ========================================================
    st.markdown("<div class='section-title'>What the data is saying</div>", unsafe_allow_html=True)
    st.caption("These are descriptive signals from Yahoo Finance data — not a BUY/SELL recommendation or score.")

    def status_label(v):
        if v in (None, "", "NO_DATA"): return "—"
        return str(v).replace("_", " ").title()

    # Descriptive summary cards
    cards = st.columns(5)
    cards[0].metric("Market Trend", status_label(result.get("Market Trend")))
    cards[1].metric("Momentum", status_label(result.get("Momentum Direction")))
    cards[2].metric("Revenue Trend", status_label(result.get("Revenue Direction")))
    cards[3].metric("Profit Trend", status_label(result.get("Profit Direction")))
    cards[4].metric("Debt Trend", status_label(result.get("Debt Direction")))

    mt1, mt2 = st.columns([1.2, 1.8])
    with mt1:
        st.markdown("**Market trend evidence**")
        market_rows = [
            ["Market trend", status_label(result.get("Market Trend"))],
            ["DMA alignment", result.get("DMA Alignment", "—")],
            ["Price vs 20 DMA", fmt_pct(result.get("Price vs 20 DMA %"))],
            ["Price vs 50 DMA", fmt_pct(result.get("Price vs 50 DMA %"))],
            ["Price vs 200 DMA", fmt_pct(result.get("Price vs 200 DMA %"))],
            ["1M return", fmt_pct(result.get("1M Trend Return %"))],
            ["3M return", fmt_pct(result.get("3M Trend Return %"))],
            ["6M return", fmt_pct(result.get("6M Trend Return %"))],
        ]
        st.dataframe(pd.DataFrame(market_rows, columns=["Measure", "Value"]), use_container_width=True, hide_index=True)
    with mt2:
        st.markdown("**How to interpret the market trend**")
        trend = result.get("Market Trend")
        if trend == "UPWARD":
            st.success("Upward trend: the current price is above more of the tracked moving averages, with the available DMA structure pointing upward.")
        elif trend == "DOWNWARD":
            st.warning("Downward trend: the current price is below more of the tracked moving averages, with the available DMA structure pointing downward.")
        elif trend == "MIXED":
            st.info("Mixed trend: price is above some moving averages and below others, so the available trend measures do not point in one direction.")
        else:
            st.info("Trend cannot be determined reliably from the available Yahoo price history.")
        st.caption(result.get("Market Trend Evidence", "—"))

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**What is improving / positive evidence**")
        pos = result.get("Positive Evidence", "")
        if pos and pos != "No clear positive change identified from available fields":
            for item in [x.strip() for x in pos.split("|") if x.strip()]:
                st.success("✓ " + item)
        else:
            st.info("No clear positive change identified from the available fields.")
    with p2:
        st.markdown("**What is weakening / watch items**")
        neg = result.get("Negative / Watch Evidence", "")
        if neg and neg != "No clear negative change identified from available fields":
            for item in [x.strip() for x in neg.split("|") if x.strip()]:
                st.warning("• " + item)
        else:
            st.info("No clear negative change identified from the available fields.")

    st.markdown("**Financial direction at a glance**")
    fin_direction = pd.DataFrame([
        ["Revenue", status_label(result.get("Revenue Direction")), fmt_pct(result.get("Revenue Q4 vs Q3 %")), fmt_pct(result.get("Revenue Q4 YoY %"))],
        ["Net Profit", status_label(result.get("Profit Direction")), fmt_pct(result.get("Profit Q4 vs Q3 %")), fmt_pct(result.get("Profit Q4 YoY %"))],
        ["EBITDA", status_label(result.get("EBITDA Direction")), fmt_pct(result.get("EBITDA Q4 vs Q3 %")), fmt_pct(result.get("EBITDA Q4 YoY %"))],
        ["Net Margin", status_label("INCREASING" if num(result.get("Net Margin Q4 vs Q3 Change")) > 0 else "DECREASING" if num(result.get("Net Margin Q4 vs Q3 Change")) < 0 else "MIXED"), "—" if pd.isna(num(result.get("Net Margin Q4 vs Q3 Change"))) else f"{num(result.get('Net Margin Q4 vs Q3 Change')):+.2f} pp", "—"],
        ["Debt", status_label(result.get("Debt Direction")), fmt_pct(result.get("Debt Q4 vs Q3 %")), "—"],
        ["Net Debt", status_label(result.get("Net Debt Direction")), fmt_pct(result.get("Net Debt Q4 vs Q3 %")), "—"],
        ["Operating Cash Flow", status_label(result.get("Operating CF Direction")), fmt_pct(result.get("Operating CF Q4 vs Q3 %")), "—"],
        ["Free Cash Flow", status_label(result.get("FCF Direction")), fmt_pct(result.get("FCF Q4 vs Q3 %")), "—"],
    ], columns=["Metric", "Direction", "QoQ Change", "YoY Change"])
    st.dataframe(fin_direction, use_container_width=True, hide_index=True)
    st.info("Revenue/Profit/EBITDA direction describes the multi-quarter pattern returned by Yahoo. QoQ compares the latest quarter with the immediately previous quarter; YoY compares the latest quarter with the comparable prior-year quarter when available.")

# ============================================================
# CHART
# ============================================================
with tab_chart:
    st.markdown("<div class='section-title'>Interactive price chart</div>", unsafe_allow_html=True)
    period_label = st.radio("Range", ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y", "Max"], index=5, horizontal=True)
    period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "max"}
    interval = "5m" if period_label == "1D" else "1h" if period_label == "5D" else "1d"
    ch = chart_history(symbol, period_map[period_label], interval)
    if ch.empty:
        st.warning("Yahoo Finance did not return chart data for this ticker/range.")
    else:
        ch["DMA20"] = ch["Close"].rolling(20).mean()
        ch["DMA50"] = ch["Close"].rolling(50).mean()
        ch["DMA200"] = ch["Close"].rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ch.index, open=ch["Open"], high=ch["High"], low=ch["Low"], close=ch["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=ch.index, y=ch["DMA20"], name="20 DMA", mode="lines", line=dict(width=1.4)))
        fig.add_trace(go.Scatter(x=ch.index, y=ch["DMA50"], name="50 DMA", mode="lines", line=dict(width=1.4)))
        fig.add_trace(go.Scatter(x=ch.index, y=ch["DMA200"], name="200 DMA", mode="lines", line=dict(width=1.6)))
        fig.update_layout(height=560, margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified", xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        vol = go.Figure()
        vol.add_trace(go.Bar(x=ch.index, y=ch["Volume"], name="Volume"))
        vol.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Volume", showlegend=False)
        st.plotly_chart(vol, use_container_width=True)

        st.download_button("⬇️ Download chart data CSV", ch.to_csv().encode("utf-8"), file_name=f"{symbol.replace('.', '_')}_{period_label}.csv", mime="text/csv")

# ============================================================
# STATISTICS
# ============================================================
with tab_statistics:
    st.markdown("<div class='section-title'>Valuation measures</div>", unsafe_allow_html=True)
    valuation = pd.DataFrame([
        ["Market Cap", fmt_crore(q["market_cap"])],
        ["Trailing P/E", fmt_num(first_value(info, ["trailingPE"]))],
        ["Forward P/E", fmt_num(first_value(info, ["forwardPE"]))],
        ["PEG Ratio", fmt_num(first_value(info, ["pegRatio"]))],
        ["Price / Book", fmt_num(first_value(info, ["priceToBook"]))],
        ["Price / Sales", fmt_num(first_value(info, ["priceToSalesTrailing12Months"]))],
        ["Enterprise Value", fmt_crore(first_value(info, ["enterpriseValue"]))],
        ["EV / EBITDA", fmt_num(first_value(info, ["enterpriseToEbitda"]))],
        ["Dividend Yield", fmt_pct((num(first_value(info, ["dividendYield"])) * 100) if pd.notna(num(first_value(info, ["dividendYield"]))) and num(first_value(info, ["dividendYield"])) < 1 else first_value(info, ["dividendYield"]))],
    ], columns=["Metric", "Value"])
    st.dataframe(valuation, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Trading statistics</div>", unsafe_allow_html=True)
    trading = pd.DataFrame([
        ["Previous Close", fmt_price(prev, currency_symbol)],
        ["Day Range", f"{fmt_price(first_value(info, ['dayLow']), currency_symbol)} – {fmt_price(first_value(info, ['dayHigh']), currency_symbol)}"],
        ["52W Range", f"{fmt_price(q['52w_low'], currency_symbol)} – {fmt_price(q['52w_high'], currency_symbol)}"],
        ["50 Day Average", fmt_price(first_value(info, ['fiftyDayAverage']), currency_symbol)],
        ["200 Day Average", fmt_price(first_value(info, ['twoHundredDayAverage']), currency_symbol)],
        ["Average Volume", fmt_num(first_value(info, ['averageVolume']), 0)],
        ["Beta", fmt_num(first_value(info, ['beta']))],
        ["Shares Outstanding", fmt_num(first_value(info, ['sharesOutstanding']), 0)],
    ], columns=["Metric", "Value"])
    st.dataframe(trading, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Company information</div>", unsafe_allow_html=True)
    company_info = pd.DataFrame([
        ["Exchange", exchange], ["Currency", currency], ["Sector", sector or "—"], ["Industry", industry or "—"],
        ["Country", first_value(info, ["country"], "—")], ["Employees", fmt_num(first_value(info, ["fullTimeEmployees"]), 0)],
    ], columns=["Field", "Value"])
    st.dataframe(company_info, use_container_width=True, hide_index=True)

# ============================================================
# FINANCIALS
# ============================================================
with tab_financials:
    frames = statements(symbol)
    fin_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

    def show_statement(df, title):
        if df is None or df.empty:
            st.warning(f"Yahoo Finance returned no {title.lower()} for this ticker.")
            return
        display = df.copy()
        display.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") if not isinstance(c, str) else c for c in display.columns]
        display.index = display.index.astype(str)
        # Convert large currency values to ₹ crore while keeping ratio/EPS rows as raw values.
        currency_rows = [
            "Total Revenue", "Operating Revenue", "EBITDA", "EBIT", "Net Income", "Net Income Common Stockholders",
            "Net Income Including Noncontrolling Interests", "Total Assets", "Stockholders Equity", "Common Stock Equity",
            "Total Debt", "Net Debt", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents",
            "Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Total Liabilities Net Minority Interest"
        ]
        for row_name in display.index:
            if row_name in currency_rows:
                display.loc[row_name] = pd.to_numeric(display.loc[row_name], errors="coerce") / 1e7
        st.dataframe(display, use_container_width=True, height=560)

    with fin_tabs[0]:
        st.caption("Annual statements · values shown in ₹ crore where applicable")
        show_statement(frames["Annual Income"], "annual income statement")
    with fin_tabs[1]:
        st.caption("Annual balance sheet · values shown in ₹ crore where applicable")
        show_statement(frames["Annual Balance Sheet"], "annual balance sheet")
    with fin_tabs[2]:
        st.caption("Annual cash flow · values shown in ₹ crore where applicable")
        show_statement(frames["Annual Cash Flow"], "annual cash flow")

# ============================================================
# QUARTERLY
# ============================================================
with tab_quarterly:
    frames = statements(symbol)

    st.markdown("<div class='section-title'>Quarter-over-quarter comparison</div>", unsafe_allow_html=True)
    st.caption("The table below uses the latest four quarters returned by Yahoo Finance. Money values are ₹ crore. The direction column tells you immediately whether the latest quarter increased or decreased versus the previous quarter.")

    qrows = []
    # Q1 is the oldest displayed quarter and Q4 is the latest.
    qdates = [result.get("Quarter -4 Date"), result.get("Quarter -3 Date"), result.get("Previous Quarter Date"), result.get("Latest Quarter Date")]
    for label, keys in [
        ("Revenue", ["Revenue Q1", "Revenue Q2", "Revenue Q3", "Revenue Q4"]),
        ("Net Profit", ["Profit Q1", "Profit Q2", "Profit Q3", "Profit Q4"]),
        ("EBITDA", ["EBITDA Q1", "EBITDA Q2", "EBITDA Q3", "EBITDA Q4"]),
        ("Net Margin", ["Net Margin Q1 %", "Net Margin Q2 %", "Net Margin Q3 %", "Net Margin Q4 %"]),
        ("Debt", ["Debt Q1", "Debt Q2", "Debt Q3", "Debt Q4"]),
        ("Cash", ["Cash Q1", "Cash Q2", "Cash Q3", "Cash Q4"]),
        ("Net Debt", ["Net Debt Q1", "Net Debt Q2", "Net Debt Q3", "Net Debt Q4"]),
        ("Operating Cash Flow", ["Operating CF Q1", "Operating CF Q2", "Operating CF Q3", "Operating CF Q4"]),
        ("Free Cash Flow", ["FCF Q1", "FCF Q2", "FCF Q3", "FCF Q4"]),
    ]:
        vals=[]
        for key in keys:
            v=num(result.get(key))
            if label == "Net Margin":
                vals.append("—" if pd.isna(v) else f"{v:.2f}%")
            else:
                vals.append("—" if pd.isna(v) else f"₹{v/1e7:,.2f} Cr")
        qrows.append([label, *vals])
    qtrend = pd.DataFrame(qrows, columns=["Metric", *[d or "Quarter" for d in qdates]])
    st.dataframe(qtrend, use_container_width=True, hide_index=True)
    ebitda_source = result.get("EBITDA Q4 Source", "—")
    if ebitda_source != "—":
        st.caption(f"EBITDA validation source for latest quarter: **{ebitda_source}**. The analytical EBITDA is not blindly copied from Yahoo's Normalized EBITDA row when that row is an extreme outlier relative to revenue.")

    qt1, qt2, qt3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    with qt1:
        st.markdown("**Validated income statement — latest four quarters**")
        q_dates = [result.get("Quarter -4 Date"), result.get("Quarter -3 Date"), result.get("Previous Quarter Date"), result.get("Latest Quarter Date")]
        valid_rows = []
        for label, keys, money in [
            ("Revenue", ["Revenue Q1","Revenue Q2","Revenue Q3","Revenue Q4"], True),
            ("Net Profit", ["Profit Q1","Profit Q2","Profit Q3","Profit Q4"], True),
            ("EBITDA", ["EBITDA Q1","EBITDA Q2","EBITDA Q3","EBITDA Q4"], True),
            ("Net Margin", ["Net Margin Q1 %","Net Margin Q2 %","Net Margin Q3 %","Net Margin Q4 %"], False),
        ]:
            vals=[]
            for key in keys:
                v=num(result.get(key))
                vals.append("—" if pd.isna(v) else (f"₹{v/1e7:,.2f} Cr" if money else f"{v:.2f}%"))
            valid_rows.append([label,*vals])
        valid_df=pd.DataFrame(valid_rows, columns=["Metric",*[d or "Quarter" for d in q_dates]])
        st.dataframe(valid_df, use_container_width=True, hide_index=True)
        st.caption(f"Validated EBITDA source by latest quarter: **{result.get('EBITDA Q4 Source','—')}**. Yahoo raw EBITDA is used only when plausible; an extreme outlier can be replaced analytically by Operating Income + D&A from Yahoo line items.")
        with st.expander("Show raw Yahoo quarterly income statement"):
            show_df = frames["Quarterly Income"]
            if show_df.empty:
                st.warning("No quarterly income statement returned by Yahoo Finance.")
            else:
                show_df = show_df.copy()
                show_df.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") if not isinstance(c, str) else c for c in show_df.columns]
                for row_name in show_df.index:
                    if row_name in ["Total Revenue", "Operating Revenue", "EBITDA", "EBIT", "Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests", "Normalized EBITDA"]:
                        show_df.loc[row_name] = pd.to_numeric(show_df.loc[row_name], errors="coerce") / 1e7
                st.caption("Raw Yahoo values are shown as supplied; they are not silently corrected. Use the validated table above for the analytical EBITDA figure.")
                st.dataframe(show_df, use_container_width=True, height=560)

    with qt2:
        show_df = frames["Quarterly Balance Sheet"]
        if show_df.empty:
            st.warning("No quarterly balance sheet returned by Yahoo Finance.")
        else:
            show_df = show_df.copy()
            show_df.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") if not isinstance(c, str) else c for c in show_df.columns]
            for row_name in show_df.index:
                show_df.loc[row_name] = pd.to_numeric(show_df.loc[row_name], errors="coerce") / 1e7
            st.dataframe(show_df, use_container_width=True, height=560)
    with qt3:
        show_df = frames["Quarterly Cash Flow"]
        if show_df.empty:
            st.warning("No quarterly cash flow returned by Yahoo Finance.")
        else:
            show_df = show_df.copy()
            show_df.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") if not isinstance(c, str) else c for c in show_df.columns]
            for row_name in show_df.index:
                show_df.loc[row_name] = pd.to_numeric(show_df.loc[row_name], errors="coerce") / 1e7
            st.dataframe(show_df, use_container_width=True, height=560)

# ============================================================
# NEWS
# ============================================================
with tab_news:
    st.markdown("<div class='section-title'>Yahoo Finance news</div>", unsafe_allow_html=True)
    news_rows = normalize_news_items(company_news(symbol))
    if not news_rows:
        st.info("Yahoo Finance did not return company news for this ticker.")
    else:
        category_filter = st.multiselect(
            "Filter by topic",
            sorted({c for n in news_rows for c in classify_news(n["text"])}),
            default=[],
        )
        shown=0
        for n in news_rows:
            cats=classify_news(n["text"])
            if category_filter and not any(c in category_filter for c in cats):
                continue
            date_text=n["datetime"].strftime("%d %b %Y %H:%M") if n["datetime"] else "Date unavailable"
            st.markdown(f"**{n['title']}**")
            st.caption(f"{n['publisher']} · {date_text} · {', '.join(cats)}")
            if n["summary"]:
                st.write(n["summary"][:500])
            if n["link"]:
                st.markdown(f"[Open source article]({n['link']})")
            shown += 1
            if shown >= 15:
                break
        if shown == 0:
            st.info("No news matched the selected topics.")

# ============================================================
# RAW DATA
# ============================================================
with tab_raw:
    st.markdown("<div class='section-title'>Analyzer record</div>", unsafe_allow_html=True)
    raw = pd.DataFrame([result]).T.reset_index()
    raw.columns = ["Field", "Value"]
    st.dataframe(raw, use_container_width=True, height=620, hide_index=True)
    st.download_button("⬇️ Download analyzer record CSV", raw.to_csv(index=False).encode("utf-8"), file_name=f"{symbol.replace('.', '_')}_analysis.csv", mime="text/csv")

st.divider()
st.caption("Built from the uploaded Stock_Market_Analyzer notebook. Data source: Yahoo Finance via yfinance. This dashboard presents facts and calculated metrics; it does not issue BUY/SELL recommendations or rankings.")
