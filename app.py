# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from scipy.stats import skew, kurtosis, norm, probplot, jarque_bera
import math

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Comparison and Analysis App")
st.caption("Compare 2 to 5 stocks, benchmark them against the S&P 500, and explore diversification.")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker_text = st.sidebar.text_input(
    "Enter 2 to 5 stock tickers (comma-separated)",
    value="AAPL,MSFT,NVDA"
)

default_start = date.today() - timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("Please select at least a 1-year date range.")
    st.stop()

vol_window = st.sidebar.selectbox("Rolling Volatility Window (days)", [30, 60, 90], index=1)
roll_corr_window = st.sidebar.selectbox("Rolling Correlation Window (days)", [30, 60, 90], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("About / Methodology")
st.sidebar.write(
    "This app uses adjusted close prices from Yahoo Finance via yfinance. "
    "Returns are simple arithmetic returns from pct_change(). "
    "Annualized return = mean daily return × 252. "
    "Annualized volatility = daily standard deviation × √252."
)

tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))

if len(tickers) < 2:
    st.error("Please enter at least 2 ticker symbols.")
    st.stop()

if len(tickers) > 5:
    st.error("Please enter no more than 5 ticker symbols.")
    st.stop()

benchmark = "^GSPC"

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers: list[str], start: date, end: date):
    all_tickers = tickers + [benchmark]
    price_series = []
    failed = []

    for t in all_tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty or "Adj Close" not in df.columns:
                failed.append(t)
                continue
            s = df["Adj Close"].copy()
            s.name = t
            price_series.append(s)
        except Exception:
            failed.append(t)

    if not price_series:
        return None, failed, []

    prices = pd.concat(price_series, axis=1)

    missing_pct = prices.isna().mean()
    dropped = [c for c in prices.columns if c != benchmark and missing_pct[c] > 0.05]
    kept = [c for c in prices.columns if c not in dropped]
    prices = prices[kept]

    prices = prices.dropna()

    if prices.empty:
        return None, failed, dropped

    return prices, failed, dropped

# -- Main logic -------------------------------------------
try:
    prices, failed_tickers, dropped_tickers = load_data(tickers, start_date, end_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

if prices is None or prices.empty:
    st.error("No usable data was found for your selected tickers.")
    st.stop()

if failed_tickers:
    st.warning(f"These tickers failed to download or had insufficient data: {', '.join(failed_tickers)}")

if dropped_tickers:
    st.warning(f"These tickers were dropped for having more than 5% missing data: {', '.join(dropped_tickers)}")

stock_cols = [c for c in prices.columns if c != benchmark]

if len(stock_cols) < 2:
    st.error("After cleaning the data, fewer than 2 valid stocks remained. Try different tickers.")
    st.stop()

returns = prices.pct_change().dropna()
stock_returns = returns[stock_cols]
benchmark_returns = returns[benchmark]

summary = pd.DataFrame(index=returns.columns)
summary["Annualized Mean Return"] = returns.mean() * 252
summary["Annualized Volatility"] = returns.std() * np.sqrt(252)
summary["Skewness"] = returns.apply(skew)
summary["Kurtosis"] = returns.apply(kurtosis)
summary["Min Daily Return"] = returns.min()
summary["Max Daily Return"] = returns.max()

equal_weight_returns = stock_returns.mean(axis=1)
wealth = (1 + returns).cumprod() * 10000
wealth["Equal-Weight Portfolio"] = (1 + equal_weight_returns).cumprod() * 10000

tab1, tab2, tab3, tab4 = st.tabs([
    "Prices & Returns",
    "Risk & Distribution",
    "Correlation",
    "Portfolio Explorer"
])

with tab1:
    st.subheader("Adjusted Closing Prices")
    selected_price_stocks = st.multiselect(
        "Select stocks to show",
        stock_cols,
        default=stock_cols
    )

    if selected_price_stocks:
        fig_price = px.line(
            prices[selected_price_stocks],
            x=prices.index,
            y=selected_price_stocks,
            title="Adjusted Closing Prices",
            labels={"value": "Adjusted Close", "index": "Date", "variable": "Ticker"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(summary.style.format({
        "Annualized Mean Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Skewness": "{:.3f}",
        "Kurtosis": "{:.3f}",
        "Min Daily Return": "{:.2%}",
        "Max Daily Return": "{:.2%}",
    }), use_container_width=True)

    st.subheader("Growth of $10,000")
    wealth_cols = stock_cols + [benchmark, "Equal-Weight Portfolio"]
    fig_wealth = px.line(
        wealth[wealth_cols],
        x=wealth.index,
        y=wealth_cols,
        title="Cumulative Wealth Index",
        labels={"value": "Portfolio Value ($)", "index": "Date", "variable": "Series"}
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

with tab2:
    st.subheader("Rolling Annualized Volatility")
    rolling_vol = stock_returns.rolling(vol_window).std() * np.sqrt(252)
    fig_vol = px.line(
        rolling_vol,
        x=rolling_vol.index,
        y=rolling_vol.columns,
        title=f"{vol_window}-Day Rolling Annualized Volatility",
        labels={"value": "Volatility", "index": "Date", "variable": "Ticker"}
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Distribution Analysis")
    selected_stock = st.selectbox("Select a stock", stock_cols)
    dist_mode = st.radio("Choose plot", ["Histogram", "Q-Q Plot"], horizontal=True)

    selected_returns = stock_returns[selected_stock].dropna()
    jb_stat, jb_pvalue = jarque_bera(selected_returns)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.metric("Jarque-Bera", f"{jb_stat:.3f}")
        st.metric("p-value", f"{jb_pvalue:.5f}")
        if jb_pvalue < 0.05:
            st.error("Rejects normality (p < 0.05)")
        else:
            st.success("Fails to reject normality (p ≥ 0.05)")

    with col1:
        if dist_mode == "Histogram":
            mu, sigma = norm.fit(selected_returns)
            x_vals = np.linspace(selected_returns.min(), selected_returns.max(), 300)
            pdf_vals = norm.pdf(x_vals, mu, sigma)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=selected_returns,
                histnorm="probability density",
                name="Daily Returns"
            ))
            fig_hist.add_trace(go.Scatter(
                x=x_vals,
                y=pdf_vals,
                mode="lines",
                name="Fitted Normal Curve"
            ))
            fig_hist.update_layout(
                title=f"Histogram of Daily Returns: {selected_stock}",
                xaxis_title="Daily Return",
                yaxis_title="Density"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            qq = probplot(selected_returns, dist="norm")
            theoretical = qq[0][0]
            ordered = qq[0][1]
            slope = qq[1][0]
            intercept = qq[1][1]

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical,
                y=ordered,
                mode="markers",
                name="Q-Q Points"
            ))
            fig_qq.add_trace(go.Scatter(
                x=theoretical,
                y=slope * theoretical + intercept,
                mode="lines",
                name="Reference Line"
            ))
            fig_qq.update_layout(
                title=f"Q-Q Plot: {selected_stock}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            st.plotly_chart(fig_qq, use_container_width=True)

    st.subheader("Box Plot of Daily Returns")
    box_df = stock_returns.melt(var_name="Ticker", value_name="Daily Return")
    fig_box = px.box(
        box_df,
        x="Ticker",
        y="Daily Return",
        title="Daily Return Distributions by Stock"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Correlation Heatmap")
    corr_matrix = stock_returns.corr()
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_midpoint=0,
        title="Pairwise Correlation Matrix of Daily Returns",
        labels={"color": "Correlation"}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Scatter Plot")
    col_a, col_b = st.columns(2)
    with col_a:
        stock_a_scatter = st.selectbox("Select Stock A", stock_cols, key="scatter_a")
    with col_b:
        stock_b_scatter = st.selectbox(
            "Select Stock B",
            [s for s in stock_cols if s != stock_a_scatter],
            key="scatter_b"
        )

    scatter_df = stock_returns[[stock_a_scatter, stock_b_scatter]].dropna()
    fig_scatter = px.scatter(
        scatter_df,
        x=stock_a_scatter,
        y=stock_b_scatter,
        title=f"{stock_a_scatter} vs {stock_b_scatter} Daily Returns"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Correlation")
    rolling_corr = stock_returns[stock_a_scatter].rolling(roll_corr_window).corr(stock_returns[stock_b_scatter])
    fig_corr = px.line(
        x=rolling_corr.index,
        y=rolling_corr.values,
        title=f"{roll_corr_window}-Day Rolling Correlation: {stock_a_scatter} vs {stock_b_scatter}",
        labels={"x": "Date", "y": "Correlation"}
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader("Two-Asset Portfolio Explorer")

    col1, col2 = st.columns(2)
    with col1:
        stock_a = st.selectbox("Choose Stock A", stock_cols, key="port_a")
    with col2:
        stock_b = st.selectbox("Choose Stock B", [s for s in stock_cols if s != stock_a], key="port_b")

    weight_a_pct = st.slider("Weight on Stock A (%)", 0, 100, 50)
    w = weight_a_pct / 100
    pair_returns = stock_returns[[stock_a, stock_b]].dropna()

    ann_mean = pair_returns.mean() * 252
    ann_cov = pair_returns.cov() * 252

    port_return = w * ann_mean[stock_a] + (1 - w) * ann_mean[stock_b]
    port_var = (
        (w ** 2) * ann_cov.loc[stock_a, stock_a]
        + ((1 - w) ** 2) * ann_cov.loc[stock_b, stock_b]
        + 2 * w * (1 - w) * ann_cov.loc[stock_a, stock_b]
    )
    port_vol = np.sqrt(port_var)

    m1, m2 = st.columns(2)
    m1.metric("Portfolio Annualized Return", f"{port_return:.2%}")
    m2.metric("Portfolio Annualized Volatility", f"{port_vol:.2%}")

    weights = np.linspace(0, 1, 101)
    vols = []

    for wt in weights:
        var = (
            (wt ** 2) * ann_cov.loc[stock_a, stock_a]
            + ((1 - wt) ** 2) * ann_cov.loc[stock_b, stock_b]
            + 2 * wt * (1 - wt) * ann_cov.loc[stock_a, stock_b]
        )
        vols.append(np.sqrt(var))

    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(
        x=weights,
        y=vols,
        mode="lines",
        name="Portfolio Volatility Curve"
    ))
    fig_port.add_trace(go.Scatter(
        x=[w],
        y=[port_vol],
        mode="markers",
        name="Current Weight"
    ))
    fig_port.update_layout(
        title=f"Portfolio Volatility vs Weight on {stock_a}",
        xaxis_title=f"Weight on {stock_a}",
        yaxis_title="Annualized Volatility"
    )
    st.plotly_chart(fig_port, use_container_width=True)

    st.info(
        "This chart shows diversification. When correlation is less than 1, "
        "combining two stocks can reduce portfolio volatility below the volatility of either stock alone."
    )