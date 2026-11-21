# Automatic A-Grade Swing Trade Scanner (Streamlit)

A Streamlit dashboard that downloads daily US equity history through `yfinance`,
calculates SMA20/50/200, RSI(14), relative volume, ATR-based risk guides, and
ranks a curated universe to surface the top three technical candidates.

## Easiest way to use it without installing Python locally

Deploy it to Streamlit Community Cloud:

1. Create/sign in to a GitHub account.
2. Create a new repository (public is simplest).
3. Upload `app.py` and `requirements.txt` to the repository root.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository, branch (`main`), and entrypoint `app.py`.
6. Deploy. Community Cloud installs the packages in `requirements.txt`.
7. Open the resulting `*.streamlit.app` URL on your computer or phone.

Official instructions:
- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Run locally (optional)

Requires Python 3.10+.

```bash
python -m venv .venv
```

Windows:
```bat
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important limitations

- The default universe is 100 selected, liquid, widely followed US equities. It
  is not a full-market scan of every NYSE/Nasdaq listing.
- Price data comes from Yahoo Finance through the third-party `yfinance` library.
  It is not an official real-time feed; it can be delayed, incomplete, rate-limited,
  or unavailable.
- This app uses daily bars, not live intraday data.
- Earnings dates, breaking news, SEC filings, bid/ask spreads, and upcoming events
  are not part of the grade. Check them manually before trading.
- "A grade" is a rules-based technical screen, not a guarantee or a buy signal.
  If no symbols pass all gates, the app explicitly reports no A-grade setups.

## Scoring summary

100-point technical score:
- Trend alignment: 25
- RSI/momentum: 15
- Relative volume: 15
- Pullback/breakout structure: 20
- Average dollar liquidity: 10
- ATR/swing-low stop-distance suitability: 15

A grade requires a score >= 82 and all hard technical gates.
