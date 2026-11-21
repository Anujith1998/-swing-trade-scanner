from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
import time as time_module

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="A-Grade Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Liquid, widely followed US-listed stocks. This is a practical scan universe,
# not every stock listed on NYSE/Nasdaq.
UNIVERSE_100 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","ORCL","CRM","AMD",
    "INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","ADI","ANET","CSCO",
    "IBM","NOW","PLTR","PANW","CRWD","FTNT","DDOG","SNPS","CDNS","NFLX",
    "DIS","COST","WMT","TGT","HD","LOW","TJX","NKE","MCD","SBUX",
    "KO","PEP","PG","CL","JPM","BAC","WFC","GS","MS","C",
    "AXP","V","MA","SCHW","BLK","COIN","HOOD","XOM","CVX","COP",
    "SLB","GE","CAT","DE","HON","ETN","RTX","LMT","NOC","UNH",
    "LLY","JNJ","ABBV","MRK","AMGN","GILD","ISRG","ABT","TMO","DHR",
    "ABNB","UBER","DASH","TSLA","F","GM","RBLX","APP","SHOP","SNOW",
    "NET","PANW","MELI","BKNG","ADBE","PYPL","XYZ","SOFI","CEG","VRT",
]
UNIVERSE_50 = UNIVERSE_100[:50]

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 12% 0%, #142b49 0, #08111f 42%);}
[data-testid="stHeader"] {background: rgba(8,17,31,.78);}
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.hero {padding:1.25rem 1.4rem;border:1px solid #294866;border-radius:18px;
background:linear-gradient(115deg,#132944,#101c30);margin-bottom:1rem;}
.hero h1 {margin:0;color:#f3f7ff;font-size:2.1rem;}
.hero p {margin:.35rem 0 0;color:#b4c7df;}
.note {padding:.85rem 1rem;border-radius:12px;background:#1b2b42;
border:1px solid #355374;color:#dce9f8;margin-bottom:.8rem;}
.good {color:#70edb0;font-weight:800;}
.bad {color:#ff8994;font-weight:800;}
.metric-card {background:#111f33;border:1px solid #2b4564;border-radius:14px;padding:1rem;}
.candidate {background:linear-gradient(145deg,#142741,#101b2d);border:1px solid #355577;
border-radius:16px;padding:1rem 1.1rem;margin-bottom:.7rem;}
.candidate h3 {margin:0 0 .2rem;color:#f4f8ff;}
.muted {color:#a9bbd2;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 Automatic A-Grade Swing Trade Scanner</h1>
  <p>US equities · 7-calendar-day holding horizon · Daily technicals · Ranked top 3 candidates</p>
</div>
<div class="note"><b>Data note:</b> Uses Yahoo Finance historical daily prices through the third-party
<code>yfinance</code> package. Data may be delayed, incomplete, rate-limited, or unavailable.
This scans a selected liquid-stock universe, not every US-listed ticker. Always verify the live quote,
earnings date, company news, spreads, and market conditions before trading.</div>
""", unsafe_allow_html=True)

def normalize_tickers(text: str) -> list[str]:
    raw = re.split(r"[\s,;]+", text.upper().strip())
    out, seen = [], set()
    for ticker in raw:
        ticker = ticker.strip()
        if not ticker:
            continue
        if not re.fullmatch(r"[A-Z0-9.^=_-]{1,15}", ticker):
            continue
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out

@st.cache_data(ttl=900, show_spinner=False)
def download_market_data(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Download 1 year of daily OHLCV in manageable batches."""
    downloaded: dict[str, pd.DataFrame] = {}
    batch_size = 15
    for start in range(0, len(tickers), batch_size):
        batch = list(tickers[start:start + batch_size])
        raw = yf.download(
            tickers=batch,
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=False,
            progress=False,
            timeout=25,
            multi_level_index=True,
        )
        if raw is None or raw.empty:
            continue

        for ticker in batch:
            frame = None
            if isinstance(raw.columns, pd.MultiIndex):
                for level in range(raw.columns.nlevels):
                    try:
                        level_values = raw.columns.get_level_values(level)
                        if ticker in level_values:
                            candidate = raw.xs(ticker, axis=1, level=level, drop_level=True)
                            if isinstance(candidate.columns, pd.MultiIndex):
                                candidate.columns = candidate.columns.get_level_values(-1)
                            if {"Open", "High", "Low", "Close", "Volume"}.issubset(candidate.columns):
                                frame = candidate.copy()
                                break
                    except (KeyError, ValueError):
                        pass
            else:
                if len(batch) == 1 and {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
                    frame = raw.copy()

            if frame is not None:
                frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if not frame.empty:
                    downloaded[ticker] = frame
        time_module.sleep(0.25)
    return downloaded

def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["DOLLARVOL20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    previous_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["PRIOR20_HIGH"] = df["High"].shift(1).rolling(20).max()
    df["LOW10"] = df["Low"].rolling(10).min()
    return df.dropna()

def analyze_ticker(ticker: str, frame: pd.DataFrame) -> dict | None:
    # Validate the raw one-year history BEFORE indicator warm-up rows are dropped.
    # SMA200, RSI, ATR and rolling-volume calculations naturally remove the
    # first ~200 rows, leaving only the most recent several dozen valid rows.
    raw_history = frame.dropna(subset=["Close"])
    if len(raw_history) < 205:
        return None

    df = compute_indicators(frame)
    if len(df) < 20:
        return None

    # Exclude an in-progress US daily candle before 4:00 p.m. ET.
    now_et = datetime.now(ZoneInfo("America/New_York"))
    last_idx = df.index[-1]
    try:
        last_date = pd.Timestamp(last_idx).date()
    except Exception:
        last_date = now_et.date()
    if last_date == now_et.date() and now_et.time() < time(16, 0) and len(df) > 205:
        df = df.iloc[:-1]

    # Indicator calculations have already removed the SMA200 warm-up period;
    # two rows are enough for the latest close and prior-session comparison.
    if len(df) < 2:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(row["Close"])
    sma20, sma50, sma200 = (float(row[k]) for k in ["SMA20", "SMA50", "SMA200"])
    rsi = float(row["RSI14"])
    atr = float(row["ATR14"])
    avg_vol = float(row["VOL20"])
    dollar_vol = float(row["DOLLARVOL20"])
    rvol = float(row["Volume"] / avg_vol) if avg_vol > 0 else 0.0
    atr_pct = atr / price * 100 if price > 0 else math.inf

    # Setup classification: pullback to the 20-day average, or a breakout above
    # the previous 20-session high. These are candidate triggers, not orders.
    pullback_near = (
        price >= sma20 * 0.98
        and price <= sma20 * 1.025
        and price > sma50
        and price > sma200
        and sma20 > sma50
        and (price > float(prev["Close"]) or float(row["Close"]) >= float(row["Open"]))
    )
    breakout = price > float(row["PRIOR20_HIGH"]) and rvol >= 1.3
    if pullback_near:
        setup = "20-day trend pullback"
    elif breakout:
        setup = "20-session breakout"
    elif price > sma20 and price > sma50 and price > sma200:
        setup = "Uptrend; waiting for trigger"
    else:
        setup = "No confirmed setup"

    score = 0
    reasons = []
    trend_ok = price > sma50 and price > sma200 and sma20 > sma50
    if trend_ok:
        score += 25
        reasons.append("Trend aligned above SMA50/SMA200")
    elif price > sma200 and sma20 > sma50:
        score += 16
        reasons.append("Longer trend improving")
    elif price > sma200:
        score += 10
        reasons.append("Price above SMA200 only")
    else:
        reasons.append("Trend not aligned")

    if 50 <= rsi <= 68:
        score += 15
        reasons.append("RSI in preferred 50–68 zone")
    elif 45 <= rsi <= 72:
        score += 10
        reasons.append("RSI acceptable, not ideal")
    else:
        reasons.append("RSI outside preferred zone")

    if rvol >= 1.5:
        score += 15
        reasons.append("Relative volume ≥1.5×")
    elif rvol >= 1.2:
        score += 12
        reasons.append("Relative volume ≥1.2×")
    elif rvol >= 1.0:
        score += 6
        reasons.append("Volume near average")
    else:
        reasons.append("Volume below average")

    if pullback_near or breakout:
        score += 20
        reasons.append("Recognized pullback/breakout trigger")
    elif price > sma20 and price > sma50:
        score += 10
        reasons.append("Uptrend, but no clean entry trigger")
    else:
        reasons.append("No clear entry structure")

    if dollar_vol >= 50_000_000:
        score += 10
        reasons.append("Strong dollar liquidity")
    elif dollar_vol >= 20_000_000:
        score += 8
        reasons.append("Adequate dollar liquidity")
    else:
        reasons.append("Below preferred dollar liquidity")

    recent_support = float(df["LOW10"].iloc[-1])
    atr_stop = price - 1.5 * atr
    stop = max(atr_stop, recent_support * 0.995)
    risk_per_share = price - stop
    risk_pct = (risk_per_share / price * 100) if price > 0 and risk_per_share > 0 else math.inf
    target_2r = price + 2 * risk_per_share if math.isfinite(risk_per_share) else np.nan

    if 0 < risk_pct <= 4.0:
        score += 15
        reasons.append("Stop distance within preferred range")
    elif risk_pct <= 5.5:
        score += 12
        reasons.append("Stop distance acceptable")
    elif risk_pct <= 7.0:
        score += 7
        reasons.append("Stop distance wide")
    else:
        reasons.append("Stop distance too wide")

    hard_gates = (
        trend_ok
        and 48 <= rsi <= 70
        and rvol >= 1.15
        and (pullback_near or breakout)
        and dollar_vol >= 20_000_000
        and 0 < risk_pct <= 5.5
    )
    grade = "A" if score >= 82 and hard_gates else (
        "B" if score >= 68 else ("C" if score >= 52 else "D")
    )
    return {
        "Ticker": ticker,
        "Score": int(score),
        "Grade": grade,
        "Setup": setup,
        "Price": price,
        "RSI": rsi,
        "RelVol": rvol,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA200": sma200,
        "AvgDollarVol": dollar_vol,
        "ATR%": atr_pct,
        "StopGuide": stop,
        "Target2R": target_2r,
        "Risk%": risk_pct,
        "Date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        "Reasons": reasons,
        "History": df.tail(100)[["Close", "SMA20", "SMA50", "SMA200"]].copy(),
    }

with st.sidebar:
    st.header("Scanner settings")
    universe_choice = st.radio("Stock universe", ["Top 50 liquid stocks", "Top 100 liquid stocks", "Custom tickers"], index=1)
    if universe_choice == "Top 50 liquid stocks":
        tickers = UNIVERSE_50
    elif universe_choice == "Top 100 liquid stocks":
        tickers = UNIVERSE_100
    else:
        custom = st.text_area("Enter tickers separated by commas", "NVDA, AMD, ANET, AAPL, MSFT")
        tickers = normalize_tickers(custom)
    tickers = list(dict.fromkeys(tickers))
    st.caption(f"{len(tickers)} unique tickers selected")
    st.divider()
    st.markdown("**A-grade rules**")
    st.caption("Score ≥82/100 plus trend, RSI, relative-volume, liquidity, risk, and confirmed setup gates.")
    st.caption("Minimum planned reward/risk: 2:1, based on the technical stop guide.")
    st.divider()
    st.caption("Data source: Yahoo Finance via yfinance. Not an official real-time feed.")

col1, col2, col3 = st.columns([1.25, 1, 1])
with col1:
    run_scan = st.button("🔎 Scan for top 3", type="primary", use_container_width=True)
with col2:
    clear_cache = st.button("↻ Refresh data", use_container_width=True)
with col3:
    st.metric("Holding period", "Up to 7 days")

if clear_cache:
    download_market_data.clear()
    st.session_state.pop("scan_results", None)
    st.rerun()

if run_scan:
    if not tickers:
        st.error("Enter at least one valid ticker.")
    else:
        results = []
        progress = st.progress(0, text="Downloading daily market data…")
        status = st.empty()
        try:
            frames = download_market_data(tuple(tickers))
            status.write(f"Daily histories returned for {len(frames)} of {len(tickers)} symbols. Calculating technicals…")
            if not frames:
                st.error("Yahoo Finance returned no usable price histories. This may be a temporary provider or rate-limit issue. Wait a few minutes, then press Refresh data and scan again.")
            for i, (ticker, frame) in enumerate(frames.items(), start=1):
                try:
                    item = analyze_ticker(ticker, frame)
                    if item:
                        results.append(item)
                except Exception:
                    continue
                progress.progress(i / max(len(frames), 1), text=f"Analysing {i}/{len(frames)} histories…")
            results.sort(key=lambda x: (x["Score"], x["RelVol"]), reverse=True)
            if frames and not results:
                st.warning("Price histories downloaded, but none passed the minimum history/indicator checks. Try again later or use a smaller/custom ticker list.")
            st.session_state["scan_results"] = results
            st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            progress.empty()
            status.empty()
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Market-data request failed: {exc}")
            st.info("Yahoo may temporarily rate-limit requests. Wait a little, then press Refresh data and scan again.")

results = st.session_state.get("scan_results", [])
if results:
    scan_time = st.session_state.get("scan_time", "Unknown")
    a_candidates = [x for x in results if x["Grade"] == "A"]
    b_candidates = [x for x in results if x["Grade"] == "B"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Symbols analysed", len(results))
    m2.metric("A-grade setups", len(a_candidates))
    m3.metric("B-grade watchlist", len(b_candidates))
    m4.metric("Scan run", scan_time)
    st.subheader("🏆 Top 3 technical candidates")
    if a_candidates:
        st.success(f"{len(a_candidates)} A-grade setup(s) passed all technical gates.")
    else:
        st.warning("No A-grade setup passed all rules. Do not force a trade. The cards below are ranked watchlist candidates, not buy signals.")
    top3 = results[:3]
    cards = st.columns(3)
    for col, item, rank in zip(cards, top3, range(1, 4)):
        with col:
            st.markdown(f"""
            <div class="candidate">
              <h3>#{rank} {item['Ticker']} &nbsp; <span class="{'good' if item['Grade']=='A' else 'bad' if item['Grade'] in ['C','D'] else ''}">{item['Grade']} · {item['Score']}/100</span></h3>
              <div class="muted">{item['Setup']} · Daily data through {item['Date']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Last close", f"${item['Price']:.2f}")
            st.write(f"**RSI:** {item['RSI']:.1f} · **Rel. volume:** {item['RelVol']:.2f}×")
            st.write(f"**Stop guide:** ${item['StopGuide']:.2f}")
            st.write(f"**2R target guide:** ${item['Target2R']:.2f}")
            st.write(f"**Risk to stop:** {item['Risk%']:.2f}%")
            with st.expander("Why this ranked here"):
                for reason in item["Reasons"]:
                    st.write("• " + reason)

    st.subheader("Full ranked scan")
    view = pd.DataFrame([{k: v for k, v in x.items() if k not in ["Reasons", "History"]} for x in results])
    display_cols = ["Ticker","Score","Grade","Setup","Price","RSI","RelVol","SMA20","SMA50","SMA200","StopGuide","Target2R","Risk%","Date"]
    st.dataframe(
        view[display_cols].style.format({
            "Price":"${:.2f}", "SMA20":"${:.2f}", "SMA50":"${:.2f}", "SMA200":"${:.2f}",
            "StopGuide":"${:.2f}", "Target2R":"${:.2f}", "RSI":"{:.1f}", "RelVol":"{:.2f}×", "Risk%":"{:.2f}%"
        }),
        use_container_width=True, hide_index=True
    )
    chosen = st.selectbox("Chart a ticker", [x["Ticker"] for x in results[:min(20, len(results))]])
    selected = next(x for x in results if x["Ticker"] == chosen)
    st.subheader(f"{chosen} · Daily trend")
    st.line_chart(selected["History"], height=300)
    csv = view[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download scan results CSV", csv, file_name="swing_scan_results.csv", mime="text/csv")
else:
    st.info("Press **Scan for top 3** to fetch daily prices, calculate indicators, and rank candidates.")
    st.markdown("""
    **What qualifies as A-grade?**
    - Price above SMA50 and SMA200, with SMA20 above SMA50.
    - RSI in a controlled momentum zone (48–70).
    - Relative volume at least 1.15× its 20-session average.
    - A recognized 20-day pullback or 20-session breakout.
    - Average daily dollar turnover of at least $20 million.
    - Stop-distance estimate no wider than 5.5%.
    - Composite technical score of at least 82/100.
    """)

st.divider()
st.caption(
    "Educational screening tool only. It does not verify earnings dates, breaking news, SEC filings, "
    "borrow availability, or live bid/ask spreads. Yahoo Finance data access via yfinance is unofficial "
    "and can be delayed, incomplete, or rate-limited. A technical A grade is not a probability of profit "
    "or a recommendation to buy. Confirm price action and events before any trade."
)
