"""
==============================================================================
Discounted Cash Flow (DCF) Analysis Web App
==============================================================================

This Streamlit app automatically fetches financial data for any publicly traded
company (via Yahoo Finance) and performs a full DCF valuation.

HOW DCF WORKS (step by step):
1. Fetch historical Free Cash Flow (FCF) from the company's financials
2. Estimate a growth rate for future FCFs (based on historical trends)
3. Project FCFs forward 5-10 years
4. Calculate a "terminal value" (the company's value beyond the projection)
5. Discount ALL future cash flows back to today using WACC
6. Sum them up → that's the company's intrinsic (estimated) value
7. Divide by shares outstanding → intrinsic value per share

WACC (Weighted Average Cost of Capital):
- Blends the cost of equity (what shareholders expect) and cost of debt
- Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium  (CAPM model)
- WACC = (E/V × Re) + (D/V × Rd × (1 - Tax Rate))
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
from plotly.subplots import make_subplots


# =============================================================================
# SECTION 1: DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)  # Cache data for 1 hour to avoid redundant API calls
def fetch_company_data(ticker: str) -> dict:
    """
    Fetches all required financial data from Yahoo Finance.

    Returns a dictionary with:
        - info: Company metadata (name, beta, shares outstanding, etc.)
        - income_stmt: Annual income statements
        - balance_sheet: Annual balance sheets
        - cashflow: Annual cash flow statements
        - history: Historical stock price data
    """
    stock = yf.Ticker(ticker)

    return {
        "info": stock.info,
        "income_stmt": stock.income_stmt,         # Columns = years (most recent first)
        "balance_sheet": stock.balance_sheet,
        "cashflow": stock.cashflow,
        "history": stock.history(period="5y"),     # 5 years of price data
    }


# =============================================================================
# SECTION 1b: RISK-FREE RATE (Live 10-Year Treasury Yield from FRED)
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_risk_free_rate() -> tuple[float | None, str]:
    """
    Fetches the current 10-Year US Treasury yield as the risk-free rate.

    Strategy (try in order):
        1. FRED API — Federal Reserve Economic Data (series DGS10)
           Uses the public CSV endpoint which requires no API key.
        2. Yahoo Finance — ^TNX ticker (CBOE 10-Year Treasury Note Yield)
           Used as a fallback if FRED is unreachable.

    Returns:
        A tuple of (yield_as_decimal, source_label).
        yield_as_decimal is None if both sources fail.
    """
    import requests

    # ---- Attempt 1: Yahoo Finance ^TNX (fallback) ----
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            latest_yield = float(hist["Close"].dropna().iloc[-1]) / 100
            if 0 < latest_yield < 0.20:
                return latest_yield, "Yahoo Finance (^TNX)"
    except Exception:
        pass

    # ---- Attempt 2: FRED public CSV endpoint (no API key required) ----
    # The DGS10 series is the market yield on 10-Year Treasury constant maturity
    # We request the last 10 observations to handle weekends/holidays
    try:
        fred_url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open+sans"
            "&graph_bgcolor=%23ffffff&mode=fred&recession_bars=on"
            "&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0"
            "&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes"
            "&id=DGS10&scale=left&cosd=2020-01-01&line_color=%234572a7"
            "&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2"
            "&vintage_date=&revision_date=&nd=&ost=-99999&oet=99999"
            "&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01"
        )
        resp = requests.get(fred_url, timeout=5)
        resp.raise_for_status()

        # Parse the CSV: columns are DATE, DGS10
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        # FRED uses "." for missing values
        df = df[df["DGS10"] != "."]
        df["DGS10"] = df["DGS10"].astype(float)

        if not df.empty:
            latest_yield = float(df["DGS10"].iloc[-1]) / 100  # Convert % to decimal
            if 0 < latest_yield < 0.20:
                return latest_yield, "FRED (DGS10)"
    except Exception:
        pass  # Fall through to Yahoo Finance

    

    return None, "unavailable"


# =============================================================================
# SECTION 2: FREE CASH FLOW (FCF) EXTRACTION
# =============================================================================

def get_historical_fcf(cashflow: pd.DataFrame) -> pd.Series:
    """
    Extracts Free Cash Flow from the cash flow statement.

    FCF = Operating Cash Flow - Capital Expenditures

    Yahoo Finance labels:
        - "Operating Cash Flow" or "Total Cash From Operating Activities"
        - "Capital Expenditure" (usually negative, so we add it)
    """
    # Try different label names (Yahoo Finance isn't always consistent)
    ocf_labels = ["Operating Cash Flow", "Total Cash From Operating Activities"]
    capex_labels = ["Capital Expenditure", "Capital Expenditures"]

    ocf = None
    capex = None

    for label in ocf_labels:
        if label in cashflow.index:
            ocf = cashflow.loc[label]
            break

    for label in capex_labels:
        if label in cashflow.index:
            capex = cashflow.loc[label]
            break

    # Also check for pre-computed Free Cash Flow
    if "Free Cash Flow" in cashflow.index:
        return cashflow.loc["Free Cash Flow"].dropna().sort_index()

    if ocf is not None and capex is not None:
        # CapEx is typically negative, so OCF + CapEx = OCF - |CapEx|
        fcf = ocf + capex
        return fcf.dropna().sort_index()

    return pd.Series(dtype=float)


def calculate_fcf_growth_rate(fcf_series: pd.Series) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR) of FCF.

    CAGR = (ending_value / beginning_value) ^ (1 / num_years) - 1

    Falls back to median year-over-year growth if CAGR can't be computed
    (e.g., if beginning FCF is negative).
    """
    if len(fcf_series) < 2:
        return 0.05  # Default 5% if not enough data

    beginning = fcf_series.iloc[0]
    ending = fcf_series.iloc[-1]
    n_years = len(fcf_series) - 1

    # CAGR only works when both values are positive
    if beginning > 0 and ending > 0:
        cagr = (ending / beginning) ** (1 / n_years) - 1
        return cagr

    # Fallback: median of year-over-year percentage changes
    yoy_growth = fcf_series.pct_change().dropna()
    if len(yoy_growth) > 0:
        return float(yoy_growth.median())

    return 0.05  # Safe default


# =============================================================================
# SECTION 3: WACC CALCULATION
# =============================================================================

def calculate_wacc(
    info: dict,
    balance_sheet: pd.DataFrame,
    income_stmt: pd.DataFrame,
    risk_free_rate: float,
    equity_risk_premium: float,
) -> dict:
    """
    Calculates the Weighted Average Cost of Capital (WACC).

    Steps:
        1. Cost of Equity (Re) via CAPM = Risk-Free Rate + Beta × ERP
        2. Cost of Debt (Rd) = Interest Expense / Total Debt
        3. Tax Rate = Income Tax / Pre-Tax Income
        4. Market Cap (E) and Total Debt (D) → weights
        5. WACC = (E/V × Re) + (D/V × Rd × (1 - Tax))

    Returns a dict with all intermediate values for transparency.
    """
    # --- Step 1: Cost of Equity ---
    beta = info.get("beta", 1.0) or 1.0  # Default to market beta if missing
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # --- Step 2: Cost of Debt ---
    # Pull most recent year's data (first column in Yahoo's format)
    interest_expense = 0
    total_debt = 0

    # Interest Expense
    for label in ["Interest Expense", "Interest Expense Non Operating"]:
        if label in income_stmt.index:
            val = income_stmt.loc[label].iloc[0]
            if pd.notna(val):
                interest_expense = abs(float(val))
                break

    # Total Debt
    for label in ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]:
        if label in balance_sheet.index:
            val = balance_sheet.loc[label].iloc[0]
            if pd.notna(val):
                total_debt = abs(float(val))
                break

    cost_of_debt = (interest_expense / total_debt) if total_debt > 0 else 0.04

    # --- Step 3: Tax Rate ---
    tax_rate = 0.21  # Default US corporate rate
    if "Tax Provision" in income_stmt.index and "Pretax Income" in income_stmt.index:
        tax = income_stmt.loc["Tax Provision"].iloc[0]
        pretax = income_stmt.loc["Pretax Income"].iloc[0]
        if pd.notna(tax) and pd.notna(pretax) and pretax > 0:
            tax_rate = float(tax) / float(pretax)
            tax_rate = max(0, min(tax_rate, 0.5))  # Clamp to reasonable range

    # --- Step 4: Capital Structure Weights ---
    market_cap = float(info.get("marketCap", 0) or 0)
    equity_value = market_cap if market_cap > 0 else 1e9  # Fallback
    total_value = equity_value + total_debt

    weight_equity = equity_value / total_value
    weight_debt = total_debt / total_value

    # --- Step 5: WACC ---
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))

    return {
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "beta": beta,
        "tax_rate": tax_rate,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "total_debt": total_debt,
        "market_cap": equity_value,
    }


# =============================================================================
# SECTION 4: DCF VALUATION ENGINE
# =============================================================================

def run_dcf(
    last_fcf: float,
    growth_rate: float,
    terminal_growth_rate: float,
    wacc: float,
    projection_years: int,
    shares_outstanding: float,
    net_debt: float,
) -> dict:
    """
    Performs the core DCF calculation.

    Steps:
        1. Project future FCFs using the growth rate
        2. Calculate terminal value using the Gordon Growth Model:
           TV = Final_FCF × (1 + g) / (WACC - g)
        3. Discount all cash flows back to present value:
           PV = FCF / (1 + WACC)^year
        4. Enterprise Value = sum of all discounted cash flows
        5. Equity Value = Enterprise Value - Net Debt
        6. Intrinsic Value Per Share = Equity Value / Shares Outstanding

    Returns a dict with projected FCFs, their present values, and the final valuation.
    """
    # --- Step 1: Project future FCFs ---
    projected_fcfs = []
    fcf = last_fcf
    for year in range(1, projection_years + 1):
        fcf *= (1 + growth_rate)
        projected_fcfs.append({"year": year, "fcf": fcf})

    # --- Step 2: Terminal Value (Gordon Growth Model) ---
    # This captures all value beyond our projection period
    final_fcf = projected_fcfs[-1]["fcf"]
    if wacc <= terminal_growth_rate:
        # Safety: WACC must exceed terminal growth, otherwise model breaks
        terminal_value = final_fcf * 20  # Simple fallback multiple
    else:
        terminal_value = final_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    # --- Step 3: Discount everything to present value ---
    pv_fcfs = []
    total_pv_fcfs = 0
    for item in projected_fcfs:
        discount_factor = (1 + wacc) ** item["year"]
        pv = item["fcf"] / discount_factor
        pv_fcfs.append({
            "year": item["year"],
            "fcf": item["fcf"],
            "pv": pv,
            "discount_factor": discount_factor,
        })
        total_pv_fcfs += pv

    pv_terminal = terminal_value / ((1 + wacc) ** projection_years)

    # --- Step 4 & 5: Enterprise Value → Equity Value ---
    enterprise_value = total_pv_fcfs + pv_terminal
    equity_value = enterprise_value - net_debt  # Subtract debt, add cash

    # --- Step 6: Per-share value ---
    intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        "projected_fcfs": pv_fcfs,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "total_pv_fcfs": total_pv_fcfs,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "net_debt": net_debt,
    }


# =============================================================================
# SECTION 5: SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(
    last_fcf: float,
    base_growth: float,
    base_wacc: float,
    terminal_growth: float,
    projection_years: int,
    shares_outstanding: float,
    net_debt: float,
    growth_range: float = 0.03,
    growth_step: float = 0.01,
    wacc_range: float = 0.02,
    wacc_step: float = 0.005,
) -> pd.DataFrame:
    """
    Builds a sensitivity table showing how the intrinsic value changes
    across different WACC and growth rate assumptions.

    This is critical because small changes in assumptions → big valuation swings.

    Parameters:
        growth_range: How far above/below the base growth rate to go (e.g., 0.03 = ±3pp)
        growth_step:  Step size between growth rate columns (e.g., 0.01 = 1pp)
        wacc_range:   How far above/below the base WACC to go (e.g., 0.02 = ±2pp)
        wacc_step:    Step size between WACC rows (e.g., 0.005 = 0.5pp)

    Defaults produce a ~7-column × ~9-row table with realistic granularity:
        - WACC varies ±2% in 0.50pp steps → captures real-world WACC uncertainty
        - Growth varies ±3% in 1pp steps → covers bear to bull scenarios
    """
    # Build the axis values symmetrically around the base case
    # np.arange can have float rounding issues, so we round to 6 decimals
    n_growth_steps = int(round(growth_range / growth_step))
    n_wacc_steps = int(round(wacc_range / wacc_step))

    growth_rates = [
        round(base_growth + i * growth_step, 6)
        for i in range(-n_growth_steps, n_growth_steps + 1)
    ]
    waccs = [
        round(base_wacc + i * wacc_step, 6)
        for i in range(-n_wacc_steps, n_wacc_steps + 1)
    ]

    # Safety: floor growth at -10%, floor WACC at 3%
    growth_rates = [max(g, -0.10) for g in growth_rates]
    waccs = [max(w, 0.03) for w in waccs]

    # Build the table: columns = growth rates, rows = WACCs
    results = {}
    for g in growth_rates:
        col_values = []
        for w in waccs:
            dcf = run_dcf(last_fcf, g, terminal_growth, w, projection_years, shares_outstanding, net_debt)
            col_values.append(dcf["intrinsic_value_per_share"])
        results[f"Growth {g:.1%}"] = col_values

    df = pd.DataFrame(results, index=[f"WACC {w:.2%}" for w in waccs])
    return df


# =============================================================================
# SECTION 5b: MONTE CARLO SIMULATION
# =============================================================================

def run_monte_carlo(
    last_fcf: float,
    base_growth: float,
    base_wacc: float,
    base_terminal_growth: float,
    projection_years: int,
    shares_outstanding: float,
    net_debt: float,
    fcf_series: pd.Series,
    num_simulations: int = 10000,
) -> dict:
    """
    Runs a Monte Carlo simulation to generate a probability distribution
    of intrinsic values.

    HOW IT WORKS:
        1. Estimate uncertainty ranges for each key input:
           - FCF Growth Rate: std dev from historical year-over-year FCF volatility
           - WACC: ±1.5% standard deviation (reflects rate/beta uncertainty)
           - Terminal Growth Rate: ±0.5% standard deviation
        2. For each of N simulations, randomly sample each input from a
           normal distribution centered on the base-case estimate.
        3. Run a full DCF for each sample → collect the intrinsic value per share.
        4. Analyze the resulting distribution: mean, median, percentiles,
           and the probability that the stock is undervalued.

    Returns a dict with the raw simulation results and summary statistics.
    """
    rng = np.random.default_rng(seed=42)  # Reproducible results

    # ---- Step 1: Determine input uncertainty (standard deviations) ----

    # Growth rate std dev: based on historical FCF volatility
    # If FCF history is volatile, the growth estimate is less certain
    yoy_changes = fcf_series.pct_change().dropna()
    if len(yoy_changes) >= 2:
        growth_std = float(yoy_changes.std())
        # Clamp: at least 2% uncertainty, at most 15%
        growth_std = max(0.02, min(growth_std, 0.15))
    else:
        growth_std = 0.05  # Default 5% if not enough history

    wacc_std = 0.015           # ±1.5% reflects beta/rate uncertainty
    terminal_growth_std = 0.005  # ±0.5% (terminal growth is tightly bounded)

    # ---- Step 2: Sample inputs for each simulation ----
    growth_samples = rng.normal(base_growth, growth_std, num_simulations)
    wacc_samples = rng.normal(base_wacc, wacc_std, num_simulations)
    tg_samples = rng.normal(base_terminal_growth, terminal_growth_std, num_simulations)

    # Apply constraints to keep the model from blowing up
    growth_samples = np.clip(growth_samples, -0.15, 0.50)   # -15% to +50%
    wacc_samples = np.clip(wacc_samples, 0.03, 0.25)        # 3% to 25%
    tg_samples = np.clip(tg_samples, 0.005, 0.045)          # 0.5% to 4.5%

    # Ensure WACC > terminal growth (Gordon Growth Model requirement)
    # Add a minimum 0.5% spread
    tg_samples = np.minimum(tg_samples, wacc_samples - 0.005)

    # ---- Step 3: Run DCF for each simulation (vectorized for speed) ----
    # Instead of calling run_dcf() in a Python loop 10K+ times, we compute
    # all simulations simultaneously using NumPy broadcasting.
    #
    # For each simulation i:
    #   projected FCFs: fcf_year_n[i] = last_fcf × (1 + g[i])^n
    #   terminal value: tv[i] = fcf_final[i] × (1 + tg[i]) / (wacc[i] - tg[i])
    #   discount factors: df_n[i] = (1 + wacc[i])^n
    #   PV of FCFs: sum over years of fcf_year_n[i] / df_n[i]
    #   PV of TV: tv[i] / (1 + wacc[i])^projection_years
    #   enterprise value: PV_fcfs[i] + PV_tv[i]
    #   equity value: EV[i] - net_debt
    #   intrinsic per share: equity[i] / shares

    years = np.arange(1, projection_years + 1)  # [1, 2, ..., N]

    # Shape: (num_simulations, projection_years)
    # Each row = one simulation, each column = one projected year
    growth_matrix = (1 + growth_samples[:, np.newaxis]) ** years[np.newaxis, :]
    fcf_matrix = last_fcf * growth_matrix  # Projected FCFs

    # Discount factors: (1 + wacc)^year for each sim × year combo
    discount_matrix = (1 + wacc_samples[:, np.newaxis]) ** years[np.newaxis, :]

    # Present value of each year's FCF
    pv_matrix = fcf_matrix / discount_matrix
    total_pv_fcfs = pv_matrix.sum(axis=1)  # Sum across years → shape (num_simulations,)

    # Terminal value (Gordon Growth Model)
    final_fcf = fcf_matrix[:, -1]  # Last projected year's FCF
    terminal_values = final_fcf * (1 + tg_samples) / (wacc_samples - tg_samples)

    # PV of terminal value
    pv_terminal = terminal_values / ((1 + wacc_samples) ** projection_years)

    # Enterprise Value → Equity Value → Per Share
    enterprise_values = total_pv_fcfs + pv_terminal
    equity_values = enterprise_values - net_debt
    intrinsic_values = equity_values / shares_outstanding if shares_outstanding > 0 else equity_values

    # ---- Step 4: Compute summary statistics ----
    # Filter out extreme outliers (beyond 1st/99th percentile) for cleaner stats
    p1, p99 = np.percentile(intrinsic_values, [1, 99])
    filtered = intrinsic_values[(intrinsic_values >= p1) & (intrinsic_values <= p99)]

    stats = {
        "mean": float(np.mean(filtered)),
        "median": float(np.median(filtered)),
        "std": float(np.std(filtered)),
        "p5": float(np.percentile(intrinsic_values, 5)),
        "p10": float(np.percentile(intrinsic_values, 10)),
        "p25": float(np.percentile(intrinsic_values, 25)),
        "p50": float(np.percentile(intrinsic_values, 50)),
        "p75": float(np.percentile(intrinsic_values, 75)),
        "p90": float(np.percentile(intrinsic_values, 90)),
        "p95": float(np.percentile(intrinsic_values, 95)),
    }

    return {
        "intrinsic_values": intrinsic_values,
        "filtered_values": filtered,
        "stats": stats,
        "num_simulations": num_simulations,
        "growth_std": growth_std,
        "wacc_std": wacc_std,
        "terminal_growth_std": terminal_growth_std,
    }


def create_monte_carlo_chart(
    mc_results: dict,
    current_price: float,
    base_intrinsic: float,
    ticker: str,
) -> go.Figure:
    """
    Creates an interactive histogram of Monte Carlo simulation results.

    Key visual features:
        - Histogram of simulated intrinsic values
        - Vertical line at current market price (red)
        - Vertical line at base-case DCF value (green)
        - Shaded regions for bear/base/bull cases
    """
    values = mc_results["filtered_values"]
    stats = mc_results["stats"]

    fig = go.Figure()

    # ---- Histogram of simulated intrinsic values ----
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=80,
        marker_color="rgba(99, 110, 250, 0.6)",
        marker_line_color="rgba(99, 110, 250, 1)",
        marker_line_width=0.5,
        name="Simulated Values",
        hovertemplate="Value: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
    ))

    # ---- Vertical reference lines ----
    y_max = len(values) / 20  # Approximate bar height for annotation placement

    # Current market price (red dashed)
    fig.add_vline(
        x=current_price, line_dash="dash", line_color="#EF553B", line_width=2,
        annotation_text=f"Market Price: ${current_price:,.2f}",
        annotation_position="top right",
        annotation_font_color="#EF553B",
    )

    # Base-case DCF value (green dashed)
    fig.add_vline(
        x=base_intrinsic, line_dash="dash", line_color="#00CC96", line_width=2,
        annotation_text=f"Base DCF: ${base_intrinsic:,.2f}",
        annotation_position="top left",
        annotation_font_color="#00CC96",
    )

    # ---- 10th–90th percentile shading ----
    fig.add_vrect(
        x0=stats["p10"], x1=stats["p90"],
        fillcolor="rgba(99, 110, 250, 0.08)",
        line_width=0,
        annotation_text="10th–90th Percentile",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="gray",
    )

    fig.update_layout(
        title=f"{ticker} — Monte Carlo DCF Simulation ({mc_results['num_simulations']:,} scenarios)",
        xaxis_title="Intrinsic Value Per Share ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=450,
        showlegend=False,
        bargap=0.02,
    )

    return fig


# =============================================================================
# SECTION 6: VISUALIZATION
# =============================================================================

def create_valuation_chart(dcf_results: dict, current_price: float, ticker: str) -> go.Figure:
    """Creates an interactive Plotly chart showing the DCF waterfall breakdown."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Projected FCF & Present Values", "Value Breakdown"),
        column_widths=[0.55, 0.45],
    )

    years = [item["year"] for item in dcf_results["projected_fcfs"]]
    fcfs = [item["fcf"] / 1e9 for item in dcf_results["projected_fcfs"]]
    pvs = [item["pv"] / 1e9 for item in dcf_results["projected_fcfs"]]

    # Left chart: FCF projections vs their present values
    fig.add_trace(
        go.Bar(name="Projected FCF", x=years, y=fcfs, marker_color="#636EFA"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="Present Value", x=years, y=pvs, marker_color="#00CC96"),
        row=1, col=1,
    )

    # Right chart: Value composition
    labels = ["PV of FCFs", "PV of Terminal Value", "− Net Debt", "Equity Value"]
    values = [
        dcf_results["total_pv_fcfs"] / 1e9,
        dcf_results["pv_terminal"] / 1e9,
        -dcf_results["net_debt"] / 1e9,
        dcf_results["equity_value"] / 1e9,
    ]
    colors = ["#636EFA", "#00CC96", "#EF553B", "#AB63FA"]

    fig.add_trace(
        go.Bar(name="Components", x=labels, y=values, marker_color=colors, showlegend=False),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"{ticker} — DCF Valuation Breakdown",
        height=450,
        barmode="group",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="USD (Billions)", row=1, col=1)
    fig.update_yaxes(title_text="USD (Billions)", row=1, col=2)

    return fig


# =============================================================================
# SECTION 7: STREAMLIT UI
# =============================================================================

# =============================================================================
# SECTION 8: APPENDIX — FULL STEP-BY-STEP MARKDOWN WALKTHROUGH
# =============================================================================

def generate_appendix(
    company_name: str,
    ticker: str,
    fcf_series: pd.Series,
    growth_rate: float,
    historical_growth: float,
    manual_override: bool,
    wacc_data: dict,
    risk_free_rate: float,
    equity_risk_premium: float,
    terminal_growth_rate: float,
    projection_years: int,
    dcf_results: dict,
    current_price: float,
    shares: float,
    cash: float,
    last_fcf: float,
    mc_results: dict = None,
) -> str:
    """
    Generates a full, step-by-step DCF walkthrough in Markdown format.

    Every intermediate value is shown with its formula so the user can
    follow (and reproduce) the entire calculation by hand.
    """
    # ---- Helpers for clean number formatting inside the markdown ----
    def fmt(n):
        """Format large numbers readably."""
        if abs(n) >= 1e12:
            return f"${n/1e12:,.2f}T"
        elif abs(n) >= 1e9:
            return f"${n/1e9:,.2f}B"
        elif abs(n) >= 1e6:
            return f"${n/1e6:,.2f}M"
        else:
            return f"${n:,.2f}"

    def fmtn(n):
        """Format without dollar sign."""
        if abs(n) >= 1e12:
            return f"{n/1e12:,.2f}T"
        elif abs(n) >= 1e9:
            return f"{n/1e9:,.2f}B"
        elif abs(n) >= 1e6:
            return f"{n/1e6:,.2f}M"
        else:
            return f"{n:,.2f}"

    # ---- Build the markdown string section by section ----
    md = ""

    # ==== HEADER ====
    md += f"# 📑 DCF Appendix — {company_name} ({ticker})\n\n"
    md += "This appendix walks through every calculation performed in the DCF model, "
    md += "showing the exact formulas and numbers used at each step.\n\n"
    md += "---\n\n"

    # ==== STEP 1: HISTORICAL FCF ====
    md += "## Step 1: Historical Free Cash Flow (FCF)\n\n"
    md += "**Formula:** `FCF = Operating Cash Flow − Capital Expenditures`\n\n"
    md += "Historical FCF values pulled from the cash flow statement:\n\n"
    md += "| Fiscal Year | Free Cash Flow |\n"
    md += "|:-----------:|:--------------:|\n"
    for date, value in fcf_series.items():
        md += f"| {date.strftime('%Y')} | {fmt(value)} |\n"
    md += f"\n**Most recent FCF (base year):** {fmt(last_fcf)}\n\n"
    md += "---\n\n"

    # ==== STEP 2: GROWTH RATE ====
    md += "## Step 2: FCF Growth Rate Estimation\n\n"
    md += "**Formula (CAGR):**\n\n"
    md += r"$$\text{CAGR} = \left(\frac{\text{Ending FCF}}{\text{Beginning FCF}}\right)^{\frac{1}{n}} - 1$$"
    md += "\n\n"

    beginning_fcf = fcf_series.iloc[0]
    ending_fcf = fcf_series.iloc[-1]
    n_years = len(fcf_series) - 1

    md += f"- Beginning FCF ({fcf_series.index[0].strftime('%Y')}): {fmt(beginning_fcf)}\n"
    md += f"- Ending FCF ({fcf_series.index[-1].strftime('%Y')}): {fmt(ending_fcf)}\n"
    md += f"- Number of periods: {n_years}\n"
    md += f"- **Historical CAGR: {historical_growth:.2%}**\n\n"

    if manual_override:
        md += f"⚠️ *User manually overrode the growth rate to **{growth_rate:.2%}***\n\n"
    else:
        md += f"✅ *Using historical CAGR (capped to reasonable bounds): **{growth_rate:.2%}***\n\n"

    md += "---\n\n"

    # ==== STEP 3: WACC ====
    md += "## Step 3: Weighted Average Cost of Capital (WACC)\n\n"
    md += "WACC is the discount rate — it represents the blended cost of all the company's capital.\n\n"

    # Step 3a: Cost of Equity
    md += "### 3a. Cost of Equity (CAPM)\n\n"
    md += r"$$R_e = R_f + \beta \times ERP$$"
    md += "\n\n"
    md += f"| Variable | Value |\n"
    md += f"|:---------|------:|\n"
    md += f"| Risk-Free Rate (Rf) | {risk_free_rate:.2%} |\n"
    md += f"| Beta (β) | {wacc_data['beta']:.2f} |\n"
    md += f"| Equity Risk Premium (ERP) | {equity_risk_premium:.2%} |\n\n"
    md += f"**Calculation:** {risk_free_rate:.2%} + {wacc_data['beta']:.2f} × {equity_risk_premium:.2%} = **{wacc_data['cost_of_equity']:.2%}**\n\n"

    # Step 3b: Cost of Debt
    md += "### 3b. Cost of Debt\n\n"
    md += r"$$R_d = \frac{\text{Interest Expense}}{\text{Total Debt}}$$"
    md += "\n\n"
    md += f"**Cost of Debt: {wacc_data['cost_of_debt']:.2%}**\n\n"

    # Step 3c: Tax Rate
    md += "### 3c. Effective Tax Rate\n\n"
    md += r"$$\text{Tax Rate} = \frac{\text{Tax Provision}}{\text{Pretax Income}}$$"
    md += "\n\n"
    md += f"**Effective Tax Rate: {wacc_data['tax_rate']:.1%}**\n\n"

    # Step 3d: Capital Structure
    md += "### 3d. Capital Structure Weights\n\n"
    md += f"| Component | Value |\n"
    md += f"|:----------|------:|\n"
    md += f"| Market Cap (Equity, E) | {fmt(wacc_data['market_cap'])} |\n"
    md += f"| Total Debt (D) | {fmt(wacc_data['total_debt'])} |\n"
    md += f"| Total Capital (V = E + D) | {fmt(wacc_data['market_cap'] + wacc_data['total_debt'])} |\n"
    md += f"| Weight of Equity (E/V) | {wacc_data['weight_equity']:.2%} |\n"
    md += f"| Weight of Debt (D/V) | {wacc_data['weight_debt']:.2%} |\n\n"

    # Step 3e: Final WACC
    md += "### 3e. WACC Formula\n\n"
    md += r"$$WACC = \left(\frac{E}{V} \times R_e\right) + \left(\frac{D}{V} \times R_d \times (1 - T)\right)$$"
    md += "\n\n"
    we = wacc_data['weight_equity']
    wd = wacc_data['weight_debt']
    re = wacc_data['cost_of_equity']
    rd = wacc_data['cost_of_debt']
    t = wacc_data['tax_rate']
    md += f"**Calculation:** ({we:.2%} × {re:.2%}) + ({wd:.2%} × {rd:.2%} × (1 − {t:.1%}))\n\n"
    md += f"= {we * re:.4%} + {wd * rd * (1 - t):.4%}\n\n"
    md += f"### ➡️ **WACC = {wacc_data['wacc']:.2%}**\n\n"
    md += "---\n\n"

    # ==== STEP 4: FCF PROJECTIONS ====
    md += "## Step 4: Project Future Free Cash Flows\n\n"
    md += f"Starting from the base-year FCF of {fmt(last_fcf)}, each year grows by **{growth_rate:.2%}**:\n\n"
    md += r"$$FCF_n = FCF_{n-1} \times (1 + g)$$"
    md += "\n\n"
    md += "| Year | Calculation | Projected FCF |\n"
    md += "|:----:|:------------|:-------------:|\n"

    projected = dcf_results["projected_fcfs"]
    prev_fcf = last_fcf
    for item in projected:
        yr = item["year"]
        fcf_val = item["fcf"]
        md += f"| {yr} | {fmt(prev_fcf)} × (1 + {growth_rate:.2%}) | **{fmt(fcf_val)}** |\n"
        prev_fcf = fcf_val

    md += "\n---\n\n"

    # ==== STEP 5: TERMINAL VALUE ====
    md += "## Step 5: Terminal Value (Gordon Growth Model)\n\n"
    md += "The terminal value captures all cash flows beyond the projection period, "
    md += "assuming the company grows at a constant rate forever.\n\n"
    md += r"$$TV = \frac{FCF_{final} \times (1 + g_{terminal})}{WACC - g_{terminal}}$$"
    md += "\n\n"

    final_fcf = projected[-1]["fcf"]
    tv = dcf_results["terminal_value"]
    md += f"| Variable | Value |\n"
    md += f"|:---------|------:|\n"
    md += f"| Final Year FCF | {fmt(final_fcf)} |\n"
    md += f"| Terminal Growth Rate (g) | {terminal_growth_rate:.2%} |\n"
    md += f"| WACC | {wacc_data['wacc']:.2%} |\n\n"
    md += f"**Calculation:** {fmt(final_fcf)} × (1 + {terminal_growth_rate:.2%}) / ({wacc_data['wacc']:.2%} − {terminal_growth_rate:.2%})\n\n"
    md += f"### ➡️ **Terminal Value = {fmt(tv)}**\n\n"
    md += "---\n\n"

    # ==== STEP 6: DISCOUNT TO PRESENT VALUE ====
    md += "## Step 6: Discount All Cash Flows to Present Value\n\n"
    md += "Every future cash flow is discounted back to today using the WACC:\n\n"
    md += r"$$PV = \frac{FCF}{(1 + WACC)^n}$$"
    md += "\n\n"
    md += "| Year | Projected FCF | Discount Factor | Present Value |\n"
    md += "|:----:|:-------------:|:---------------:|:-------------:|\n"

    for item in projected:
        yr = item["year"]
        md += f"| {yr} | {fmt(item['fcf'])} | (1 + {wacc_data['wacc']:.2%})^{yr} = {item['discount_factor']:.4f} | **{fmt(item['pv'])}** |\n"

    pv_tv = dcf_results["pv_terminal"]
    md += f"| TV | {fmt(tv)} | (1 + {wacc_data['wacc']:.2%})^{projection_years} = {(1 + wacc_data['wacc'])**projection_years:.4f} | **{fmt(pv_tv)}** |\n"

    md += f"\n**Sum of PV of projected FCFs:** {fmt(dcf_results['total_pv_fcfs'])}\n\n"
    md += f"**PV of Terminal Value:** {fmt(pv_tv)}\n\n"
    md += "---\n\n"

    # ==== STEP 7: ENTERPRISE → EQUITY → PER SHARE ====
    md += "## Step 7: From Enterprise Value to Intrinsic Value Per Share\n\n"

    ev = dcf_results["enterprise_value"]
    nd = dcf_results["net_debt"]
    eq_val = dcf_results["equity_value"]
    ivps = dcf_results["intrinsic_value_per_share"]

    md += "### 7a. Enterprise Value\n\n"
    md += r"$$EV = \sum PV(FCFs) + PV(Terminal\ Value)$$"
    md += "\n\n"
    md += f"= {fmt(dcf_results['total_pv_fcfs'])} + {fmt(pv_tv)} = **{fmt(ev)}**\n\n"

    md += "### 7b. Net Debt\n\n"
    md += r"$$\text{Net Debt} = \text{Total Debt} - \text{Cash \& Equivalents}$$"
    md += "\n\n"
    md += f"= {fmt(wacc_data['total_debt'])} − {fmt(cash)} = **{fmt(nd)}**\n\n"
    if nd < 0:
        md += "*(Negative net debt means the company holds more cash than debt — this adds to equity value.)*\n\n"

    md += "### 7c. Equity Value\n\n"
    md += r"$$\text{Equity Value} = EV - \text{Net Debt}$$"
    md += "\n\n"
    md += f"= {fmt(ev)} − {fmt(nd)} = **{fmt(eq_val)}**\n\n"

    md += "### 7d. Intrinsic Value Per Share\n\n"
    md += r"$$\text{Intrinsic Value} = \frac{\text{Equity Value}}{\text{Shares Outstanding}}$$"
    md += "\n\n"
    md += f"= {fmt(eq_val)} / {fmtn(shares)} = **${ivps:,.2f}**\n\n"

    md += "---\n\n"

    # ==== STEP 8: VERDICT ====
    md += "## Step 8: Compare to Market Price\n\n"
    upside = ((ivps - current_price) / current_price * 100) if current_price > 0 else 0

    md += f"| Metric | Value |\n"
    md += f"|:-------|------:|\n"
    md += f"| **Intrinsic Value / Share** | **${ivps:,.2f}** |\n"
    md += f"| Current Market Price | ${current_price:,.2f} |\n"
    md += f"| Upside / Downside | {upside:+.1f}% |\n\n"

    if upside > 15:
        md += f"📗 **Potentially Undervalued** — the model suggests the stock is worth {upside:.0f}% more than its current price.\n\n"
    elif upside < -15:
        md += f"📕 **Potentially Overvalued** — the model suggests the stock is worth {abs(upside):.0f}% less than its current price.\n\n"
    else:
        md += f"📘 **Fairly Valued** — the intrinsic value is within ~15% of the market price.\n\n"

    md += "---\n\n"

    # ==== STEP 9 (OPTIONAL): MONTE CARLO SUMMARY ====
    if mc_results is not None:
        stats = mc_results["stats"]
        iv = mc_results["intrinsic_values"]
        prob_undervalued = float(np.sum(iv > current_price) / len(iv) * 100)

        md += "## Step 9: Monte Carlo Simulation\n\n"
        md += f"To quantify uncertainty, we ran **{mc_results['num_simulations']:,} simulations**, "
        md += "each with randomly sampled inputs drawn from normal distributions:\n\n"

        md += "### Input Distributions\n\n"
        md += r"$$X_i \sim \mathcal{N}(\mu, \sigma)$$"
        md += "\n\n"
        md += "| Input | Mean (μ) | Std Dev (σ) |\n"
        md += "|:------|:--------:|:-----------:|\n"
        md += f"| FCF Growth Rate | {growth_rate:.2%} | {mc_results['growth_std']:.2%} |\n"
        md += f"| WACC | {wacc_data['wacc']:.2%} | {mc_results['wacc_std']:.2%} |\n"
        md += f"| Terminal Growth | {terminal_growth_rate:.2%} | {mc_results['terminal_growth_std']:.2%} |\n\n"

        md += "### Results Summary\n\n"
        md += "| Statistic | Value |\n"
        md += "|:----------|------:|\n"
        md += f"| Mean Intrinsic Value | ${stats['mean']:,.2f} |\n"
        md += f"| Median Intrinsic Value | ${stats['median']:,.2f} |\n"
        md += f"| Standard Deviation | ${stats['std']:,.2f} |\n"
        md += f"| 5th Percentile (Bear) | ${stats['p5']:,.2f} |\n"
        md += f"| 25th Percentile | ${stats['p25']:,.2f} |\n"
        md += f"| 75th Percentile | ${stats['p75']:,.2f} |\n"
        md += f"| 95th Percentile (Bull) | ${stats['p95']:,.2f} |\n\n"

        md += f"### ➡️ **Probability stock is undervalued: {prob_undervalued:.1f}%**\n\n"
        md += f"This means that in {prob_undervalued:.0f}% of the {mc_results['num_simulations']:,} "
        md += f"simulated scenarios, the intrinsic value exceeded the current market price of ${current_price:,.2f}.\n\n"

        md += "---\n\n"

    md += "*⚠️ Disclaimer: This is an educational tool, not financial advice. "
    md += "DCF valuations are highly sensitive to input assumptions. Always conduct your own research.*\n"

    return md


def format_large_number(n: float) -> str:
    """Formats numbers into human-readable strings (e.g., 1.5T, 300B, 50M)."""
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    elif abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    else:
        return f"${n:,.0f}"


def main():
    st.set_page_config(page_title="DCF Analyzer", page_icon="📊", layout="wide")

    st.title("📊 Discounted Cash Flow (DCF) Analyzer")
    st.markdown("Automatically fetch financials and calculate the intrinsic value of any public company.")

    # ---- Sidebar: User Inputs ----
    with st.sidebar:
        st.header("⚙️ Parameters")

        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter any valid Yahoo Finance ticker symbol").upper().strip()

        st.subheader("Assumptions")

        # --- Risk-Free Rate: auto-fetch from FRED/Treasury, let user override ---
        live_rfr, rfr_source_name = fetch_risk_free_rate()
        if live_rfr is not None:
            rfr_default = round(live_rfr * 100, 2)  # Convert to percentage for display
            rfr_source = f"📡 Live 10Y Treasury: {rfr_default:.2f}% (via {rfr_source_name})"
        else:
            rfr_default = 4.25
            rfr_source = "⚠️ Could not fetch live rate — using default 4.25%"
        st.caption(rfr_source)

        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.5, max_value=10.0, value=rfr_default, step=0.05,
            help="10-Year US Treasury yield. Auto-fetched; adjust if needed.",
        ) / 100

        equity_risk_premium = st.slider(
            "Equity Risk Premium (%)", 3.0, 8.0, 5.5, 0.25,
            help="Extra return investors demand over the risk-free rate"
        ) / 100

        terminal_growth_rate = st.slider(
            "Terminal Growth Rate (%)", 1.0, 4.0, 2.5, 0.25,
            help="Long-term perpetual growth (usually ~GDP growth)"
        ) / 100

        projection_years = st.slider("Projection Years", 3, 15, 5)

        override_growth = st.checkbox("Override FCF growth rate?")
        manual_growth = None
        if override_growth:
            manual_growth = st.slider("Manual FCF Growth Rate (%)", -10.0, 40.0, 10.0, 0.5) / 100

        # --- Monte Carlo Controls ---
        st.divider()
        st.subheader("🎲 Monte Carlo")
        run_monte = st.checkbox("Run Monte Carlo Simulation", value=True)
        num_simulations = 10000
        if run_monte:
            num_simulations = st.select_slider(
                "Simulations",
                options=[1000, 5000, 10000, 25000, 50000],
                value=10000,
                help="More simulations = smoother distribution but slower runtime"
            )

        # --- Sensitivity Analysis Controls ---
        st.divider()
        st.subheader("🔍 Sensitivity Table")
        customize_sensitivity = st.checkbox("Customize sensitivity ranges")

        # Defaults: WACC ±2% in 0.5pp steps, Growth ±3% in 1pp steps
        sens_wacc_range = 2.0
        sens_wacc_step = 0.50
        sens_growth_range = 3.0
        sens_growth_step = 1.0

        if customize_sensitivity:
            scol1, scol2 = st.columns(2)
            with scol1:
                sens_wacc_range = st.number_input(
                    "WACC range (± pp)", min_value=0.5, max_value=5.0,
                    value=2.0, step=0.5,
                    help="How far above/below base WACC to test"
                )
                sens_wacc_step = st.number_input(
                    "WACC step (pp)", min_value=0.10, max_value=2.0,
                    value=0.50, step=0.10, format="%.2f",
                    help="Increment between WACC rows"
                )
            with scol2:
                sens_growth_range = st.number_input(
                    "Growth range (± pp)", min_value=1.0, max_value=10.0,
                    value=3.0, step=0.5,
                    help="How far above/below base growth rate to test"
                )
                sens_growth_step = st.number_input(
                    "Growth step (pp)", min_value=0.25, max_value=5.0,
                    value=1.0, step=0.25, format="%.2f",
                    help="Increment between growth rate columns"
                )

        # Convert from percentage points to decimals for the function
        sens_wacc_range_dec = sens_wacc_range / 100
        sens_wacc_step_dec = sens_wacc_step / 100
        sens_growth_range_dec = sens_growth_range / 100
        sens_growth_step_dec = sens_growth_step / 100

        run_analysis = st.button("🚀 Run DCF Analysis", type="primary", use_container_width=True)

    # ---- Main Content ----
    if run_analysis:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                data = fetch_company_data(ticker)
            except Exception as e:
                st.error(f"Could not fetch data for '{ticker}'. Verify the ticker is valid.\n\nError: {e}")
                return

        info = data["info"]
        company_name = info.get("longName", ticker)

        # ---- Company Overview (shown above tabs) ----
        st.header(f"{company_name} ({ticker})")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
        shares = info.get("sharesOutstanding", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:,.2f}")
        col2.metric("Market Cap", format_large_number(info.get("marketCap", 0)))
        col3.metric("Beta", f"{info.get('beta', 'N/A')}")
        col4.metric("Shares Outstanding", format_large_number(shares))

        # ---- Extract Historical FCF (needed by both tabs) ----
        fcf_series = get_historical_fcf(data["cashflow"])

        if fcf_series.empty or len(fcf_series) < 2:
            st.error("Not enough Free Cash Flow data available for this company.")
            return

        # ---- Growth Rate (needed by both tabs) ----
        historical_growth = calculate_fcf_growth_rate(fcf_series)
        growth_rate = manual_growth if manual_growth is not None else historical_growth
        growth_rate = max(-0.10, min(growth_rate, 0.40))  # Cap extreme rates

        # ---- WACC (needed by both tabs) ----
        wacc_data = calculate_wacc(info, data["balance_sheet"], data["income_stmt"], risk_free_rate, equity_risk_premium)

        # ---- Net Debt (needed by both tabs) ----
        cash = 0
        for label in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
            if label in data["balance_sheet"].index:
                val = data["balance_sheet"].loc[label].iloc[0]
                if pd.notna(val):
                    cash = float(val)
                    break
        net_debt = wacc_data["total_debt"] - cash

        # ---- Run DCF (needed by both tabs) ----
        last_fcf = float(fcf_series.iloc[-1])
        dcf_results = run_dcf(
            last_fcf=last_fcf,
            growth_rate=growth_rate,
            terminal_growth_rate=terminal_growth_rate,
            wacc=wacc_data["wacc"],
            projection_years=projection_years,
            shares_outstanding=shares,
            net_debt=net_debt,
        )

        intrinsic = dcf_results["intrinsic_value_per_share"]
        upside = ((intrinsic - current_price) / current_price * 100) if current_price > 0 else 0

        # ================================================================
        # TABS: Analysis | Monte Carlo | Appendix
        # ================================================================
        tab_analysis, tab_monte_carlo, tab_appendix = st.tabs([
            "📊 Analysis",
            "🎲 Monte Carlo Simulation",
            "📑 Appendix — Full Calculation",
        ])

        # ---- TAB 1: ANALYSIS (original dashboard) ----
        with tab_analysis:

            st.subheader("📈 Historical Free Cash Flow")
            fcf_display = pd.DataFrame({
                "Year": [d.strftime("%Y") for d in fcf_series.index],
                "Free Cash Flow": [format_large_number(v) for v in fcf_series.values],
            })
            st.dataframe(fcf_display, use_container_width=True, hide_index=True)

            st.info(
                f"**Historical FCF CAGR:** {historical_growth:.1%}  |  "
                f"**Using:** {growth_rate:.1%} {'(manual override)' if manual_growth is not None else '(historical)'}"
            )

            # WACC metrics
            st.subheader("💰 WACC Calculation")
            wcol1, wcol2, wcol3, wcol4 = st.columns(4)
            wcol1.metric("WACC", f"{wacc_data['wacc']:.2%}")
            wcol2.metric("Cost of Equity", f"{wacc_data['cost_of_equity']:.2%}")
            wcol3.metric("Cost of Debt", f"{wacc_data['cost_of_debt']:.2%}")
            wcol4.metric("Effective Tax Rate", f"{wacc_data['tax_rate']:.1%}")

            with st.expander("WACC Details"):
                st.markdown(f"""
| Component | Value |
|-----------|-------|
| Beta | {wacc_data['beta']:.2f} |
| Risk-Free Rate | {risk_free_rate:.2%} |
| Equity Risk Premium | {equity_risk_premium:.2%} |
| Cost of Equity (CAPM) | {risk_free_rate:.2%} + {wacc_data['beta']:.2f} × {equity_risk_premium:.2%} = **{wacc_data['cost_of_equity']:.2%}** |
| Weight of Equity | {wacc_data['weight_equity']:.1%} |
| Weight of Debt | {wacc_data['weight_debt']:.1%} |
| WACC | ({wacc_data['weight_equity']:.1%} × {wacc_data['cost_of_equity']:.2%}) + ({wacc_data['weight_debt']:.1%} × {wacc_data['cost_of_debt']:.2%} × (1 − {wacc_data['tax_rate']:.1%})) = **{wacc_data['wacc']:.2%}** |
                """)

            # Key Results
            st.subheader("🎯 DCF Valuation")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol1.metric("Intrinsic Value / Share", f"${intrinsic:,.2f}")
            rcol2.metric("Current Price", f"${current_price:,.2f}")
            rcol3.metric("Upside / Downside", f"{upside:+.1f}%", delta=f"{upside:+.1f}%")
            rcol4.metric("Enterprise Value", format_large_number(dcf_results["enterprise_value"]))

            # Verdict
            if upside > 15:
                st.success(f"📗 **Potentially Undervalued** — intrinsic value is {upside:.0f}% above market price.")
            elif upside < -15:
                st.warning(f"📕 **Potentially Overvalued** — intrinsic value is {abs(upside):.0f}% below market price.")
            else:
                st.info(f"📘 **Fairly Valued** — intrinsic value is within ~15% of market price.")

            # Chart
            fig = create_valuation_chart(dcf_results, current_price, ticker)
            st.plotly_chart(fig, use_container_width=True)

            # Projected FCF table
            with st.expander("📋 Detailed Projected Cash Flows"):
                proj_df = pd.DataFrame(dcf_results["projected_fcfs"])
                proj_df["year"] = proj_df["year"].apply(lambda y: f"Year {y}")
                proj_df["fcf"] = proj_df["fcf"].apply(lambda v: format_large_number(v))
                proj_df["pv"] = proj_df["pv"].apply(lambda v: format_large_number(v))
                proj_df["discount_factor"] = proj_df["discount_factor"].apply(lambda v: f"{v:.4f}")
                proj_df.columns = ["Year", "Projected FCF", "Present Value", "Discount Factor"]
                st.dataframe(proj_df, use_container_width=True, hide_index=True)

                st.markdown(f"""
**Terminal Value:** {format_large_number(dcf_results['terminal_value'])}  
**PV of Terminal Value:** {format_large_number(dcf_results['pv_terminal'])}  
**Net Debt:** {format_large_number(dcf_results['net_debt'])}
                """)

            # Sensitivity Analysis
            st.subheader("🔍 Sensitivity Analysis")
            st.markdown(
                f"How the intrinsic value per share changes with different assumptions. "
                f"Colors anchored to the current market price of **${current_price:,.2f}** "
                f"— 🟢 green = above market price, 🔴 red = below."
            )

            sens_df = sensitivity_analysis(
                last_fcf, growth_rate, wacc_data["wacc"],
                terminal_growth_rate, projection_years, shares, net_debt,
                growth_range=sens_growth_range_dec,
                growth_step=sens_growth_step_dec,
                wacc_range=sens_wacc_range_dec,
                wacc_step=sens_wacc_step_dec,
            )

            # ---- Color scale anchored to current market price ----
            # Red (#d4513a) when value is well below market price
            # White (#ffffff) when value equals market price
            # Green (#2da02d) when value is well above market price
            #
            # We determine how far each cell is from the market price as a
            # percentage, then linearly interpolate between the anchor colors.
            # The "full saturation" point is ±50% away from market price — 
            # beyond that, the color stays fully saturated.

            def _price_anchored_color(val):
                """Returns a CSS background-color string anchored to current_price."""
                if pd.isna(val) or current_price <= 0:
                    return ""
                # How far is this value from the market price, as a fraction?
                # +1.0 = 100% above, -1.0 = 100% below
                pct_diff = (val - current_price) / current_price
                # Clamp to [-0.5, +0.5] for color scaling, then normalize to [0, 1]
                intensity = min(abs(pct_diff) / 0.5, 1.0)

                if pct_diff >= 0:
                    # Interpolate white → green
                    r = int(255 - intensity * (255 - 45))   # 255 → 45
                    g = int(255 - intensity * (255 - 160))  # 255 → 160
                    b = int(255 - intensity * (255 - 45))   # 255 → 45
                else:
                    # Interpolate white → red
                    r = int(255 - intensity * (255 - 212))  # 255 → 212
                    g = int(255 - intensity * (255 - 81))   # 255 → 81
                    b = int(255 - intensity * (255 - 58))   # 255 → 58

                # Use dark text on light backgrounds, white text on dark backgrounds
                text_color = "#000000" if intensity < 0.7 else "#ffffff"
                return f"background-color: rgb({r},{g},{b}); color: {text_color}"

            styled = (
                sens_df.style
                .format("${:,.2f}")
                .map(_price_anchored_color)
            )
            st.dataframe(styled, use_container_width=True, height=min(38 * (len(sens_df) + 1), 600))

        # ---- TAB 2: MONTE CARLO SIMULATION ----
        mc_results = None  # Will be populated if the user enabled it
        with tab_monte_carlo:
            if not run_monte:
                st.info("Monte Carlo simulation is disabled. Enable it in the sidebar to run.")
            else:
                with st.spinner(f"Running {num_simulations:,} simulations..."):
                    mc_results = run_monte_carlo(
                        last_fcf=last_fcf,
                        base_growth=growth_rate,
                        base_wacc=wacc_data["wacc"],
                        base_terminal_growth=terminal_growth_rate,
                        projection_years=projection_years,
                        shares_outstanding=shares,
                        net_debt=net_debt,
                        fcf_series=fcf_series,
                        num_simulations=num_simulations,
                    )

                stats = mc_results["stats"]
                iv = mc_results["intrinsic_values"]
                prob_undervalued = float(np.sum(iv > current_price) / len(iv) * 100)

                # ---- Key Monte Carlo Metrics ----
                st.subheader("🎯 Probabilistic Valuation")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Median Value", f"${stats['median']:,.2f}")
                mcol2.metric("Mean Value", f"${stats['mean']:,.2f}")
                mcol3.metric("Std Deviation", f"${stats['std']:,.2f}")
                mcol4.metric(
                    "P(Undervalued)",
                    f"{prob_undervalued:.1f}%",
                    help="Probability that intrinsic value exceeds market price"
                )

                # ---- Distribution Chart ----
                mc_fig = create_monte_carlo_chart(mc_results, current_price, intrinsic, ticker)
                st.plotly_chart(mc_fig, use_container_width=True)

                # ---- Percentile Table ----
                st.subheader("📊 Value Distribution")
                st.markdown("Percentile ranges show the spread of possible intrinsic values:")

                pcol1, pcol2 = st.columns(2)

                with pcol1:
                    st.markdown(f"""
| Scenario | Percentile | Value |
|:---------|:----------:|------:|
| 🐻 Deep Bear | 5th | ${stats['p5']:,.2f} |
| 🐻 Bear | 10th | ${stats['p10']:,.2f} |
| ↘️ Below Base | 25th | ${stats['p25']:,.2f} |
| ➡️ **Median** | **50th** | **${stats['p50']:,.2f}** |
| ↗️ Above Base | 75th | ${stats['p75']:,.2f} |
| 🐂 Bull | 90th | ${stats['p90']:,.2f} |
| 🐂 Strong Bull | 95th | ${stats['p95']:,.2f} |
                    """)

                with pcol2:
                    # Probability gauge interpretation
                    if prob_undervalued > 70:
                        verdict_emoji = "📗"
                        verdict = "Strong Buy Signal"
                        verdict_detail = f"In {prob_undervalued:.0f}% of simulated scenarios, the stock's intrinsic value exceeds its current market price."
                    elif prob_undervalued > 50:
                        verdict_emoji = "📘"
                        verdict = "Slight Buy Signal"
                        verdict_detail = f"In {prob_undervalued:.0f}% of scenarios the stock is undervalued — a slight tilt in favor of buying."
                    elif prob_undervalued > 30:
                        verdict_emoji = "📙"
                        verdict = "Fairly Valued"
                        verdict_detail = f"The stock appears fairly priced, with only {prob_undervalued:.0f}% of scenarios showing undervaluation."
                    else:
                        verdict_emoji = "📕"
                        verdict = "Overvalued Signal"
                        verdict_detail = f"Only {prob_undervalued:.0f}% of scenarios show the stock as undervalued — the market price may be rich."

                    st.markdown(f"### {verdict_emoji} {verdict}")
                    st.markdown(verdict_detail)
                    st.markdown(f"""
**Interpretation guide:**
- The simulation randomized the growth rate, WACC, and terminal growth
  across {num_simulations:,} scenarios.
- The **10th–90th percentile range** (${stats['p10']:,.2f} – ${stats['p90']:,.2f})
  represents the most likely valuation band.
- Compare this range to the current price of **${current_price:,.2f}** to
  gauge margin of safety.
                    """)

                # ---- Input Assumptions Used ----
                with st.expander("🔧 Simulation Parameters"):
                    st.markdown(f"""
The Monte Carlo simulation sampled each input from a normal distribution:

| Input | Base Value | Std Deviation | Range (±2σ) |
|:------|:---------:|:------------:|:----------:|
| FCF Growth Rate | {growth_rate:.2%} | {mc_results['growth_std']:.2%} | {growth_rate - 2*mc_results['growth_std']:.2%} to {growth_rate + 2*mc_results['growth_std']:.2%} |
| WACC | {wacc_data['wacc']:.2%} | {mc_results['wacc_std']:.2%} | {wacc_data['wacc'] - 2*mc_results['wacc_std']:.2%} to {wacc_data['wacc'] + 2*mc_results['wacc_std']:.2%} |
| Terminal Growth | {terminal_growth_rate:.2%} | {mc_results['terminal_growth_std']:.2%} | {terminal_growth_rate - 2*mc_results['terminal_growth_std']:.2%} to {terminal_growth_rate + 2*mc_results['terminal_growth_std']:.2%} |

*Growth rate uncertainty is derived from historical FCF volatility.
WACC and terminal growth use fixed uncertainty bands.*
                    """)

        # ---- TAB 2: APPENDIX (full step-by-step walkthrough) ----
        with tab_appendix:
            appendix_md = generate_appendix(
                company_name=company_name,
                ticker=ticker,
                fcf_series=fcf_series,
                growth_rate=growth_rate,
                historical_growth=historical_growth,
                manual_override=(manual_growth is not None),
                wacc_data=wacc_data,
                risk_free_rate=risk_free_rate,
                equity_risk_premium=equity_risk_premium,
                terminal_growth_rate=terminal_growth_rate,
                projection_years=projection_years,
                dcf_results=dcf_results,
                current_price=current_price,
                shares=shares,
                cash=cash,
                last_fcf=last_fcf,
                mc_results=mc_results,
            )
            st.markdown(appendix_md)

        # ---- Disclaimers (shown below tabs) ----
        st.divider()
        st.caption(
            "⚠️ **Disclaimer:** This is an educational tool, not financial advice. "
            "DCF models are highly sensitive to assumptions (growth rate, WACC, terminal rate). "
            "Always do your own research and consult a financial advisor before investing."
        )


if __name__ == "__main__":
    main()
