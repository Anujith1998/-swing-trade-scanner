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
    page_title="Professional Trading Scanner",
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
    "NET","PANW","MELI","BKNG","ADBE","PYPL","SQ","SOFI","CEG","VRT",
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
    df = compute_indicators(frame)
    if len(df) < 205:
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

    if len(df) < 200:
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

@st.cache_data(ttl=180, show_spinner=False)
def download_intraday_data(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Download recent regular-session 5-minute OHLCV bars. Yahoo data is not guaranteed live."""
    downloaded: dict[str, pd.DataFrame] = {}
    batch_size = 10
    for start in range(0, len(tickers), batch_size):
        batch = list(tickers[start:start + batch_size])
        try:
            raw = yf.download(
                tickers=batch, period="5d", interval="5m", auto_adjust=True,
                group_by="ticker", threads=False, progress=False, timeout=25,
                prepost=False, multi_level_index=True,
            )
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for ticker in batch:
            frame = None
            if isinstance(raw.columns, pd.MultiIndex):
                for level in range(raw.columns.nlevels):
                    try:
                        if ticker in raw.columns.get_level_values(level):
                            candidate = raw.xs(ticker, axis=1, level=level, drop_level=True)
                            if isinstance(candidate.columns, pd.MultiIndex):
                                candidate.columns = candidate.columns.get_level_values(-1)
                            if {"Open", "High", "Low", "Close", "Volume"}.issubset(candidate.columns):
                                frame = candidate.copy()
                                break
                    except (KeyError, ValueError):
                        pass
            elif len(batch) == 1 and {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
                frame = raw.copy()
            if frame is not None:
                frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if not frame.empty:
                    downloaded[ticker] = frame
        time_module.sleep(0.2)
    return downloaded


def analyze_intraday_ticker(ticker: str, frame: pd.DataFrame, account_size: float, risk_pct: float) -> dict | None:
    """Score a long-only intraday momentum setup from 5-minute bars."""
    df = frame.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 30:
        return None

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert("America/New_York")
    df.index = idx_et
    # Analyse the latest trading session only; avoid mixing VWAP across sessions.
    latest_day = df.index[-1].date()
    session = df[df.index.date == latest_day].copy()
    if len(session) < 12:
        return None

    typical = (session["High"] + session["Low"] + session["Close"]) / 3
    session_volume = session["Volume"].clip(lower=0)
    cumulative_volume = session_volume.cumsum().replace(0, np.nan)
    session["VWAP"] = (typical * session_volume).cumsum() / cumulative_volume
    session["EMA9"] = session["Close"].ewm(span=9, adjust=False).mean()
    session["EMA20"] = session["Close"].ewm(span=20, adjust=False).mean()
    delta = session["Close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    session["RSI14"] = 100 - (100 / (1 + rs))
    session["RSI14"] = session["RSI14"].fillna(50)
    prev_close = session["Close"].shift(1)
    tr = pd.concat([(session["High"] - session["Low"]),
                    (session["High"] - prev_close).abs(),
                    (session["Low"] - prev_close).abs()], axis=1).max(axis=1)
    session["ATR14"] = tr.ewm(alpha=1/14, min_periods=5, adjust=False).mean()
    session["VOL20"] = session["Volume"].rolling(20, min_periods=5).mean()
    row = session.iloc[-1]
    price = float(row["Close"])
    vwap = float(row["VWAP"]) if pd.notna(row["VWAP"]) else price
    ema9, ema20 = float(row["EMA9"]), float(row["EMA20"])
    rsi = float(row["RSI14"])
    atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) else 0.0
    avg_vol = float(row["VOL20"]) if pd.notna(row["VOL20"]) and row["VOL20"] > 0 else 0.0
    rvol = float(row["Volume"] / avg_vol) if avg_vol else 0.0
    prior_high = float(session["High"].iloc[-7:-1].max()) if len(session) >= 7 else float(session["High"].iloc[:-1].max())
    recent_low = float(session["Low"].iloc[-6:].min())
    # Stop is placed below both the recent micro swing low and an ATR buffer.
    stop = min(recent_low, price - 1.2 * atr) if atr > 0 else recent_low
    risk_per_share = price - stop
    if not math.isfinite(risk_per_share) or risk_per_share <= 0 or price <= 0:
        return None
    target1, target2 = price + risk_per_share, price + 2 * risk_per_share
    risk_budget = account_size * (risk_pct / 100.0)
    shares_by_risk = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
    shares_by_cash = int(account_size // price) if price > 0 else 0
    shares = max(0, min(shares_by_risk, shares_by_cash))

    # Breakout uses the prior six 5-minute bars (approximately 30 minutes).
    breakout = price > prior_high
    trend_ok = price > vwap and ema9 > ema20
    score = 0
    reasons = []
    if trend_ok:
        score += 25; reasons.append("Price above session VWAP; EMA9 above EMA20")
    else:
        reasons.append("VWAP/EMA trend not fully aligned")
    if 50 <= rsi <= 70:
        score += 15; reasons.append("RSI in intraday momentum zone")
    elif 45 <= rsi < 50 or 70 < rsi <= 75:
        score += 8; reasons.append("RSI acceptable but not ideal")
    else:
        reasons.append("RSI weak or potentially overextended")
    if rvol >= 1.5:
        score += 20; reasons.append("Volume at least 1.5x recent bar average")
    elif rvol >= 1.2:
        score += 14; reasons.append("Volume at least 1.2x recent bar average")
    elif rvol >= 1.0:
        score += 7; reasons.append("Volume near recent bar average")
    else:
        reasons.append("Volume below recent bar average")
    if breakout:
        score += 20; reasons.append("Breakout above prior six-bar high")
    else:
        reasons.append("No six-bar breakout")
    if risk_per_share / price <= 0.025:
        score += 15; reasons.append("Stop distance within 2.5%")
    elif risk_per_share / price <= 0.04:
        score += 8; reasons.append("Stop distance moderately wide")
    else:
        reasons.append("Stop distance too wide for preferred intraday risk")
    if float(row["Volume"]) * price >= 100_000:
        score += 5; reasons.append("Current bar has reasonable dollar turnover")

    hard_gate = trend_ok and 48 <= rsi <= 72 and rvol >= 1.0 and breakout and risk_per_share / price <= 0.04
    grade = "A" if score >= 80 and hard_gate else ("B" if score >= 65 else ("C" if score >= 45 else "D"))
    setup = "VWAP momentum breakout" if breakout and trend_ok else ("VWAP trend; waiting for breakout" if trend_ok else "No confirmed intraday setup")
    return {
        "Ticker": ticker, "Score": int(score), "Grade": grade, "Setup": setup,
        "Price": price, "RSI": rsi, "RelVol": rvol, "VWAP": vwap,
        "EMA9": ema9, "EMA20": ema20, "EntryGuide": price, "StopGuide": stop,
        "Target1": target1, "Target2": target2, "Risk%": risk_per_share / price * 100,
        "Shares": shares, "RiskBudget": risk_budget, "DateTime": session.index[-1].strftime("%Y-%m-%d %H:%M ET"),
        "Reasons": reasons, "History": session[["Close", "VWAP", "EMA9", "EMA20"]].copy(),
    }


# Shared controls: blank price bounds mean no price restriction.
with st.sidebar:
    st.header("Scanner settings")
    universe_choice = st.radio("Stock universe", ["Top 50 liquid stocks", "Top 100 liquid stocks", "Custom tickers"], index=1)
    if universe_choice == "Top 50 liquid stocks":
        tickers = UNIVERSE_50
    elif universe_choice == "Top 100 liquid stocks":
        tickers = UNIVERSE_100
    else:
        custom = st.text_area("Enter tickers separated by commas", "NVDA, AMD, SOFI, PLTR, AAPL")
        tickers = normalize_tickers(custom)
    tickers = list(dict.fromkeys(tickers))
    st.caption(f"{len(tickers)} unique tickers selected")
    st.divider()
    st.subheader("Optional price filter (USD)")
    min_price = st.number_input("Minimum stock price", min_value=0.0, value=0.0, step=0.5, format="%.2f", help="Leave at 0 to set no minimum.")
    max_price = st.number_input("Maximum stock price", min_value=0.0, value=0.0, step=0.5, format="%.2f", help="Leave at 0 to set no maximum.")
    if min_price > 0 and max_price > 0 and min_price > max_price:
        st.error("Minimum price must be less than or equal to maximum price.")
    st.divider()
    st.subheader("Position sizing")
    account_size = st.number_input("Account size (USD)", min_value=100.0, value=1400.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.caption("Share sizing is a guide based on account risk and available cash; it does not include fees, slippage or gaps.")
    st.divider()
    st.caption("Data: Yahoo Finance via yfinance. Not an official real-time feed.")


def apply_price_filter(items: list[dict]) -> list[dict]:
    if min_price > 0 and max_price > 0 and min_price > max_price:
        return []
    return [x for x in items if (min_price <= 0 or x["Price"] >= min_price) and (max_price <= 0 or x["Price"] <= max_price)]


st.markdown("""
<div class="hero">
  <h1>📈 Professional Trading Scanner</h1>
  <p>US equities · Swing + Intraday · Configurable price range · Risk-based trade planning</p>
</div>
<div class="note"><b>Data note:</b> Daily and 5-minute historical data are provided by Yahoo Finance through yfinance.
Intraday data may be delayed, incomplete, or rate-limited. This scans only the selected ticker universe,
not every US-listed stock. Confirm live quotes, news, earnings, spreads and market conditions before trading.</div>
""", unsafe_allow_html=True)

swing_tab, intraday_tab = st.tabs(["📈 Swing Trading", "⚡ Intraday Trading"])

with swing_tab:
    st.subheader("Swing trade scanner")
    st.caption("Daily technical analysis · Existing A-grade rules retained · Up to 7-calendar-day holding horizon")
    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1:
        run_swing = st.button("🔎 Scan swing opportunities", type="primary", use_container_width=True, key="run_swing")
    with c2:
        refresh_swing = st.button("↻ Refresh swing data", use_container_width=True, key="refresh_swing")
    with c3:
        st.metric("Holding period", "Up to 7 days")

    if refresh_swing:
        download_market_data.clear()
        st.session_state.pop("swing_results", None)
        st.rerun()
    if run_swing:
        if not tickers:
            st.error("Enter at least one valid ticker.")
        elif min_price > 0 and max_price > 0 and min_price > max_price:
            st.error("Minimum price must not exceed maximum price.")
        else:
            progress = st.progress(0, text="Downloading daily market data…")
            status = st.empty()
            try:
                frames = download_market_data(tuple(tickers))
                raw_results = []
                status.write(f"Daily histories returned for {len(frames)} of {len(tickers)} symbols. Calculating technicals…")
                for i, (ticker, frame) in enumerate(frames.items(), start=1):
                    try:
                        item = analyze_ticker(ticker, frame)
                        if item:
                            raw_results.append(item)
                    except Exception:
                        continue
                    progress.progress(i / max(len(frames), 1), text=f"Analysing {i}/{len(frames)} histories…")
                filtered = apply_price_filter(raw_results)
                filtered.sort(key=lambda x: (x["Score"], x["RelVol"]), reverse=True)
                st.session_state["swing_results"] = filtered
                st.session_state["swing_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["swing_symbols_analysed"] = len(raw_results)
                progress.empty(); status.empty()
            except Exception as exc:
                progress.empty(); status.empty()
                st.error(f"Market-data request failed: {exc}")
                st.info("Yahoo may temporarily rate-limit requests. Wait, refresh data, then scan again.")

    results = st.session_state.get("swing_results", [])
    if results:
        a_candidates = [x for x in results if x["Grade"] == "A"]
        b_candidates = [x for x in results if x["Grade"] == "B"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Symbols analysed", st.session_state.get("swing_symbols_analysed", len(results)))
        m2.metric("Within price filter", len(results))
        m3.metric("A-grade setups", len(a_candidates))
        m4.metric("Scan run", st.session_state.get("swing_scan_time", "Unknown"))
        if a_candidates:
            st.success(f"{len(a_candidates)} A-grade setup(s) passed all technical gates.")
        else:
            st.warning("No A-grade setup passed all rules. Do not force a trade; lower grades are watchlist candidates, not buy signals.")
        st.subheader("Top 3 swing candidates")
        cards = st.columns(3)
        for col, item, rank in zip(cards, results[:3], range(1, 4)):
            with col:
                st.markdown(f"<div class='candidate'><h3>#{rank} {item['Ticker']} · {item['Grade']} · {item['Score']}/100</h3><div class='muted'>{item['Setup']} · {item['Date']}</div></div>", unsafe_allow_html=True)
                st.metric("Last close", f"${item['Price']:.2f}")
                st.write(f"**RSI:** {item['RSI']:.1f} · **Rel. volume:** {item['RelVol']:.2f}×")
                st.write(f"**Stop:** ${item['StopGuide']:.2f} · **2R target:** ${item['Target2R']:.2f}")
                risk_ps = max(item["Price"] - item["StopGuide"], 0)
                shares = min(int((account_size * risk_pct / 100) // risk_ps), int(account_size // item["Price"])) if risk_ps > 0 else 0
                st.write(f"**Suggested shares:** {shares} · **Risk:** {item['Risk%']:.2f}%")
                with st.expander("Why this ranked here"):
                    for reason in item["Reasons"]: st.write("• " + reason)
        st.subheader("Full swing scan")
        view = pd.DataFrame([{k: v for k, v in x.items() if k not in ["Reasons", "History"]} for x in results])
        cols = ["Ticker","Score","Grade","Setup","Price","RSI","RelVol","SMA20","SMA50","SMA200","StopGuide","Target2R","Risk%","Date"]
        st.dataframe(view[cols].style.format({"Price":"${:.2f}","SMA20":"${:.2f}","SMA50":"${:.2f}","SMA200":"${:.2f}","StopGuide":"${:.2f}","Target2R":"${:.2f}","RSI":"{:.1f}","RelVol":"{:.2f}×","Risk%":"{:.2f}%"}), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Chart a swing ticker", [x["Ticker"] for x in results[:min(20, len(results))]], key="swing_chart")
        selected = next(x for x in results if x["Ticker"] == chosen)
        st.line_chart(selected["History"], height=300)
        st.download_button("⬇️ Download swing results CSV", view[cols].to_csv(index=False).encode("utf-8"), file_name="swing_scan_results.csv", mime="text/csv")
    elif "swing_results" in st.session_state:
        st.warning("No stocks matched the current price filter and swing rules.")
    else:
        st.info("Press Scan swing opportunities. Leave both price fields at 0 to scan all available prices.")
        st.markdown("**Existing A-grade rules:** price above SMA50/SMA200, SMA20 above SMA50, RSI 48–70, relative volume ≥1.15×, pullback or breakout trigger, average dollar turnover ≥$20M, and stop distance ≤5.5%.")

with intraday_tab:
    st.subheader("Intraday momentum scanner")
    st.caption("5-minute bars · Session VWAP · EMA9/EMA20 · RSI · Relative bar volume · 30-minute breakout check")
    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1:
        run_intraday = st.button("⚡ Scan intraday opportunities", type="primary", use_container_width=True, key="run_intraday")
    with c2:
        refresh_intraday = st.button("↻ Refresh intraday data", use_container_width=True, key="refresh_intraday")
    with c3:
        st.metric("Chart interval", "5 minutes")
    if refresh_intraday:
        download_intraday_data.clear()
        st.session_state.pop("intraday_results", None)
        st.rerun()
    if run_intraday:
        if not tickers:
            st.error("Enter at least one valid ticker.")
        elif min_price > 0 and max_price > 0 and min_price > max_price:
            st.error("Minimum price must not exceed maximum price.")
        else:
            progress = st.progress(0, text="Downloading recent 5-minute bars…")
            status = st.empty()
            try:
                frames = download_intraday_data(tuple(tickers))
                raw_results = []
                status.write(f"Intraday histories returned for {len(frames)} of {len(tickers)} symbols. Analysing setups…")
                for i, (ticker, frame) in enumerate(frames.items(), start=1):
                    try:
                        item = analyze_intraday_ticker(ticker, frame, account_size, risk_pct)
                        if item: raw_results.append(item)
                    except Exception:
                        continue
                    progress.progress(i / max(len(frames), 1), text=f"Analysing {i}/{len(frames)} symbols…")
                filtered = apply_price_filter(raw_results)
                filtered.sort(key=lambda x: (x["Score"], x["RelVol"]), reverse=True)
                st.session_state["intraday_results"] = filtered
                st.session_state["intraday_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["intraday_symbols_analysed"] = len(raw_results)
                progress.empty(); status.empty()
            except Exception as exc:
                progress.empty(); status.empty()
                st.error(f"Intraday data request failed: {exc}")
                st.info("Yahoo may not return intraday bars for every symbol. Try fewer tickers or refresh later.")
    intraday_results = st.session_state.get("intraday_results", [])
    if intraday_results:
        a_count = sum(x["Grade"] == "A" for x in intraday_results)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Symbols analysed", st.session_state.get("intraday_symbols_analysed", len(intraday_results)))
        m2.metric("Within price filter", len(intraday_results))
        m3.metric("A-grade setups", a_count)
        m4.metric("Scan run", st.session_state.get("intraday_scan_time", "Unknown"))
        if a_count:
            st.success(f"{a_count} A-grade intraday setup(s) passed the technical gates.")
        else:
            st.warning("No A-grade intraday setup found. Wait for a clearer setup rather than forcing a trade.")
        st.subheader("Top 5 intraday candidates")
        for rank, item in enumerate(intraday_results[:5], start=1):
            with st.container(border=True):
                left, mid, right = st.columns([1.3, 1.2, 1.5])
                with left:
                    st.markdown(f"### #{rank} {item['Ticker']} · {item['Grade']} ({item['Score']}/100)")
                    st.write(item["Setup"])
                    st.metric("Last 5-minute close", f"${item['Price']:.2f}")
                with mid:
                    st.write(f"**Entry guide:** ${item['EntryGuide']:.2f}")
                    st.write(f"**Stop-loss guide:** ${item['StopGuide']:.2f}")
                    st.write(f"**Target 1 (1R):** ${item['Target1']:.2f}")
                    st.write(f"**Target 2 (2R):** ${item['Target2']:.2f}")
                with right:
                    st.write(f"**RSI:** {item['RSI']:.1f} · **Rel. volume:** {item['RelVol']:.2f}×")
                    st.write(f"**VWAP:** ${item['VWAP']:.2f}")
                    st.write(f"**Risk:** {item['Risk%']:.2f}% · **Shares:** {item['Shares']}")
                    st.write(f"**Planned max risk:** ${item['Shares'] * (item['Price'] - item['StopGuide']):.2f}")
                with st.expander("Scoring details"):
                    for reason in item["Reasons"]: st.write("• " + reason)
                    st.caption(f"Latest bar: {item['DateTime']}. Entry is a calculation guide, not an order or confirmation of a fill.")
        intraday_view = pd.DataFrame([{k: v for k, v in x.items() if k not in ["Reasons", "History"]} for x in intraday_results])
        st.subheader("Full intraday scan")
        display = ["Ticker","Score","Grade","Setup","Price","RSI","RelVol","VWAP","EMA9","EMA20","StopGuide","Target1","Target2","Risk%","Shares","DateTime"]
        st.dataframe(intraday_view[display].style.format({c:"${:.2f}" for c in ["Price","VWAP","EMA9","EMA20","StopGuide","Target1","Target2"]} | {"RSI":"{:.1f}","RelVol":"{:.2f}×","Risk%":"{:.2f}%"}), use_container_width=True, hide_index=True)
        chart_ticker = st.selectbox("Chart an intraday ticker", [x["Ticker"] for x in intraday_results[:min(20, len(intraday_results))]], key="intraday_chart")
        chart_item = next(x for x in intraday_results if x["Ticker"] == chart_ticker)
        st.line_chart(chart_item["History"], height=300)
        st.download_button("⬇️ Download intraday results CSV", intraday_view[display].to_csv(index=False).encode("utf-8"), file_name="intraday_scan_results.csv", mime="text/csv")
    elif "intraday_results" in st.session_state:
        st.warning("No stocks matched the current price filter and intraday rules.")
    else:
        st.info("Press Scan intraday opportunities. Intraday analysis uses 5-minute historical bars and may be delayed or incomplete.")
        st.markdown("**Intraday A-grade gates:** price above VWAP, EMA9 above EMA20, RSI 48–72, relative bar volume ≥1.0×, breakout above the prior six 5-minute bars, and stop distance ≤4%.")

st.divider()
st.caption("Educational screening tool only. Scores are rule-based rankings, not win probabilities. Yahoo Finance/yfinance is unofficial and may be delayed, incomplete or rate-limited. This scanner does not verify breaking news, SEC filings, earnings dates, spreads, halts, borrow availability or order-book liquidity. Stops do not guarantee execution prices; gaps and slippage can exceed planned risk. Confirm all trades with a live broker feed before acting.")
