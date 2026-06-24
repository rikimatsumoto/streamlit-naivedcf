# Automated Discounted Cash Flow Web App

A Python/Streamlit web app that automatically calculates the intrinsic value of any publicly traded company using a simple Discounted Cash Flow model.

## quick start

### installation via github

```bash
# Clone the repository
git clone https://github.com/rikimatsumoto/streamlit-naivedcf.git
cd streamlit-naivedcf

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## how does it work?

The app follows a standard DCF process:

1. **Data Fetching** - Pulls financials from Yahoo Finance (income statement, balance sheet, cash flows)
2. **FCF Extraction** — Calculates Free Cash Flow = Operating Cash Flow − CapEx
3. **Growth Estimation** — Computes historical CAGR (or lets you override manually)
4. **WACC Calculation** — Uses CAPM for cost of equity, blends with cost of debt
5. **DCF Engine** — Projects FCFs, calculates terminal value (Gordon Growth), discounts to present
6. **Sensitivity Analysis** — Shows how valuation changes across different assumptions
7. **Historical DCF Replay** - core backtest with 1/2/3-year forward returns

## features

- Works with any Yahoo Finance ticker (US and international)
- Adjustable assumptions via sidebar controls
- Interactive Plotly charts
- Full WACC breakdown with formula transparency
- Color-coded sensitivity table
- Handles edge cases (negative FCF, missing data, etc.)

## key assumptions to understand

| Parameter | What It Means | Typical Range |
|-----------|---------------|---------------|
| Risk-Free Rate | 10-year Treasury yield | 3–5% |
| Equity Risk Premium | Extra return for stock risk | 4–7% |
| Terminal Growth Rate | Perpetual growth after projection | 2–3% (≈ GDP) |
| Projection Years | How far to forecast FCFs | 5–10 years |
