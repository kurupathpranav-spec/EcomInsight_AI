"""
EcomInsight AI — AI-Powered E-Commerce Analytics Dashboard
Author : Pranav Kurupath
Dataset: Madhav E-Commerce Sales Dataset (Kaggle)
         https://www.kaggle.com/datasets/amitkumar209/madhav-e-commerce-sales-dataset
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcomInsight AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ═══════════════════════════════════════════════════════════════════
       DARK THEME — EcomInsight AI
       bg: #0f1117  |  surface: #1a1d26  |  text: #e8eaf0
    ═══════════════════════════════════════════════════════════════════ */

    /* ── App shell ──────────────────────────────────────────────────── */
    .stApp {
        background-color: #0f1117 !important;
    }

    /* ── Global text: always light on dark ──────────────────────────── */
    html, body,
    h1, h2, h3, h4, h5, h6,
    p, span, div, li,
    .stMarkdown, .stText,
    label,
    .stSelectbox label, .stMultiSelect label,
    .stDateInput label, .stSlider label,
    .stNumberInput label, .stTextInput label,
    .stTextArea label, .stCheckbox label,
    .stRadio label, .stFileUploader label {
        color: #e8eaf0 !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #1a1d26 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e8eaf0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2d3143 !important;
    }

    /* ── KPI cards ──────────────────────────────────────────────────── */
    .kpi-card {
        background: #1a1d26;
        border: 1px solid #2d3143;
        border-radius: 12px;
        padding: 16px 10px 14px 10px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }
    .kpi-label {
        font-size: 11px;
        color: #9ca3af !important;
        margin-bottom: 6px;
        white-space: nowrap;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: clamp(15px, 2.1vw, 23px);
        font-weight: 800;
        color: #f0f4ff !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 11px;
        color: #4f9cf9 !important;
        margin-top: 5px;
    }

    /* ── Section titles ─────────────────────────────────────────────── */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f0f4ff !important;
        margin-top: 28px;
        margin-bottom: 8px;
        border-left: 4px solid #4f9cf9;
        padding-left: 10px;
    }

    /* ── Insight / rec / warn boxes ─────────────────────────────────── */
    .insight-box {
        background: #1a2540;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #c9d8f5 !important;
        border-left: 4px solid #4f9cf9;
    }
    .rec-box {
        background: #162318;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #a7f3d0 !important;
        border-left: 4px solid #22c55e;
    }
    .warn-box {
        background: #231a10;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #fed7aa !important;
        border-left: 4px solid #f97316;
    }

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d26 !important;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        color: #4f9cf9 !important;
        border-bottom: 2px solid #4f9cf9 !important;
    }

    /* ── Dataframes & tables ────────────────────────────────────────── */
    .stDataFrame thead tr th {
        background-color: #1a1d26 !important;
        color: #e8eaf0 !important;
    }
    .stDataFrame tbody tr td {
        color: #e8eaf0 !important;
    }
    .stTable th, .stTable td {
        color: #e8eaf0 !important;
    }

    /* ── Captions ───────────────────────────────────────────────────── */
    .stCaption, small {
        color: #9ca3af !important;
    }

    /* ── Horizontal rules ───────────────────────────────────────────── */
    hr {
        border-color: #2d3143 !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button {
        background-color: #4f9cf9 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #3b82d4 !important;
    }

    /* ── Selectbox / multiselect / date input ───────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #1a1d26 !important;
        color: #e8eaf0 !important;
        border-color: #2d3143 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    orders_path  = os.path.join(base, "dataset", "Orders.csv")
    details_path = os.path.join(base, "dataset", "Details.csv")

    orders  = pd.read_csv(orders_path)
    details = pd.read_csv(details_path)

    # Normalise column names
    orders.columns  = orders.columns.str.strip()
    details.columns = details.columns.str.strip()

    # Merge on Order ID (many-to-one: details rows carry order info)
    df = details.merge(orders, on="Order ID", how="left")

    # Parse Order Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["Order Date"], inplace=True)

    # Derived time columns
    df["Month"]      = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Quarter"]    = df["Order Date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    df["Year"]       = df["Order Date"].dt.year
    df["MonthYear"]  = df["Order Date"].dt.to_period("M").astype(str)

    # Ensure numeric
    for col in ["Amount", "Profit", "Quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Profit Margin per row
    df["Profit Margin %"] = np.where(
        df["Amount"] != 0, (df["Profit"] / df["Amount"]) * 100, 0
    )

    df.reset_index(drop=True, inplace=True)
    return df


def validate_data(df):
    required = ["Order ID","Amount","Profit","Quantity","Category",
                "Sub-Category","PaymentMode","Order Date","State","City"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns after merge: {missing}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: KPI CARD
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, delta=""):
    st.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-delta">{delta}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING  (Linear Regression on monthly sales)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_sales(df, months_ahead=3):
    monthly = (
        df.groupby("MonthYear")["Amount"].sum()
        .reset_index()
        .sort_values("MonthYear")
    )
    monthly["t"] = np.arange(len(monthly))

    X = monthly[["t"]].values
    y = monthly["Amount"].values

    model = LinearRegression()
    model.fit(X, y)

    # Forecast future months
    last_t   = monthly["t"].max()
    last_period = pd.Period(monthly["MonthYear"].iloc[-1], freq="M")
    future_t = np.arange(last_t + 1, last_t + 1 + months_ahead).reshape(-1, 1)
    future_periods = [str(last_period + i) for i in range(1, months_ahead + 1)]
    future_sales   = model.predict(future_t)
    future_sales   = np.maximum(future_sales, 0)

    forecast_df = pd.DataFrame({
        "MonthYear": future_periods,
        "Amount":    future_sales,
        "Type":      "Forecast",
    })
    historical_df = monthly[["MonthYear","Amount"]].copy()
    historical_df["Type"] = "Historical"

    combined = pd.concat([historical_df, forecast_df], ignore_index=True)
    r2 = model.score(X, y)
    return combined, forecast_df, r2


# ─────────────────────────────────────────────────────────────────────────────
# AI INSIGHT GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_insights(df):
    insights = []
    recs     = []
    warns    = []

    total_sales  = df["Amount"].sum()
    total_profit = df["Profit"].sum()
    margin       = (total_profit / total_sales * 100) if total_sales else 0

    # Top / bottom category by profit margin
    cat_margin = (
        df.groupby("Category")
        .apply(lambda x: (x["Profit"].sum() / x["Amount"].sum() * 100) if x["Amount"].sum() else 0)
        .reset_index()
    )
    cat_margin.columns = ["Category","Margin"]
    best_cat  = cat_margin.loc[cat_margin["Margin"].idxmax(), "Category"]
    worst_cat = cat_margin.loc[cat_margin["Margin"].idxmin(), "Category"]

    # Top state
    state_sales = df.groupby("State")["Amount"].sum()
    top_state   = state_sales.idxmax()

    # Top payment mode
    pm_sales  = df.groupby("PaymentMode")["Amount"].sum()
    top_pm    = pm_sales.idxmax()

    # Monthly trend direction
    monthly = df.groupby("MonthYear")["Amount"].sum().sort_index()
    if len(monthly) >= 2:
        trend = "upward 📈" if monthly.iloc[-1] > monthly.iloc[0] else "downward 📉"
    else:
        trend = "stable"

    # Loss-making sub-categories
    sub_profit = df.groupby("Sub-Category")["Profit"].sum()
    loss_subs  = sub_profit[sub_profit < 0].index.tolist()

    # High-sales low-profit
    cat_stats = df.groupby("Category").agg(Sales=("Amount","sum"), Profit=("Profit","sum"))
    for cat, row in cat_stats.iterrows():
        cat_margin_val = (row["Profit"] / row["Sales"] * 100) if row["Sales"] else 0
        if row["Sales"] > total_sales * 0.25 and cat_margin_val < 5:
            warns.append(
                f"⚠️ <b>{cat}</b> has high sales ({row['Sales']:,.0f} ₹) but a very low margin ({cat_margin_val:.1f}%). "
                f"Review pricing or supplier costs."
            )

    insights.append(
        f"Overall profit margin is <b>{margin:.1f}%</b> across ₹{total_sales:,.0f} in total sales."
    )
    insights.append(
        f"<b>{best_cat}</b> is the most profitable category by margin; "
        f"<b>{worst_cat}</b> needs attention."
    )
    insights.append(
        f"<b>{top_state}</b> is the top revenue-generating state "
        f"(₹{state_sales[top_state]:,.0f})."
    )
    insights.append(
        f"<b>{top_pm}</b> is the dominant payment mode, accounting for "
        f"₹{pm_sales[top_pm]:,.0f} in sales."
    )
    insights.append(
        f"Monthly sales trend is <b>{trend}</b> over the available data period."
    )
    if loss_subs:
        insights.append(
            f"Loss-making sub-categories detected: <b>{', '.join(loss_subs[:5])}</b>."
        )

    recs.append(
        f"💡 Increase marketing spend on <b>{best_cat}</b> — it delivers the highest profit margin."
    )
    recs.append(
        f"💡 Offer loyalty incentives for <b>{top_pm}</b> users to sustain the dominant payment channel."
    )
    recs.append(
        f"💡 Focus customer acquisition efforts in <b>{top_state}</b> to capitalise on the existing demand."
    )
    if loss_subs:
        recs.append(
            f"💡 Discontinue or reprice loss-making sub-categories: <b>{', '.join(loss_subs[:3])}</b>."
        )
    recs.append(
        "💡 Run targeted promotions during historically low-sales months to smooth the revenue curve."
    )

    return insights, recs, warns


# ─────────────────────────────────────────────────────────────────────────────
# AI BUSINESS ANALYST — QUERY HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def answer_query(df, query: str) -> str:
    q = query.lower().strip()

    total_sales   = df["Amount"].sum()
    total_profit  = df["Profit"].sum()
    total_orders  = df["Order ID"].nunique()
    total_qty     = df["Quantity"].sum()
    avg_order_val = df.groupby("Order ID")["Amount"].sum().mean()
    margin        = (total_profit / total_sales * 100) if total_sales else 0

    # ── Sales / Revenue ──────────────────────────────────────────────────────
    if any(w in q for w in ["total sales","total revenue","revenue"]):
        return f"💰 Total sales revenue is **₹{total_sales:,.2f}** across {total_orders:,} unique orders."

    if any(w in q for w in ["total profit","net profit"]):
        return f"📊 Total profit is **₹{total_profit:,.2f}** (margin: {margin:.1f}%)."

    if "profit margin" in q:
        return f"📈 Overall profit margin is **{margin:.1f}%**."

    if "average order" in q or "aov" in q:
        return f"🛒 Average Order Value (AOV) is **₹{avg_order_val:,.2f}**."

    if "total orders" in q or "number of orders" in q:
        return f"📦 There are **{total_orders:,}** unique orders in the dataset."

    if "quantity" in q and "sold" in q:
        return f"📦 Total quantity sold is **{total_qty:,}** units."

    # ── Category ─────────────────────────────────────────────────────────────
    if "best category" in q or "top category" in q:
        top = df.groupby("Category")["Amount"].sum().idxmax()
        val = df.groupby("Category")["Amount"].sum().max()
        return f"🏆 Best-selling category is **{top}** with ₹{val:,.2f} in sales."

    if "category" in q and "profit" in q:
        cat_p = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in cat_p.items()]
        return "📊 Profit by Category:\n" + "\n".join(lines)

    if "category" in q:
        cat_s = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in cat_s.items()]
        return "📦 Sales by Category:\n" + "\n".join(lines)

    # ── Sub-Category ─────────────────────────────────────────────────────────
    if "sub-category" in q or "subcategory" in q:
        sub_s = df.groupby("Sub-Category")["Amount"].sum().sort_values(ascending=False).head(10)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in sub_s.items()]
        return "📦 Top 10 Sub-Categories by Sales:\n" + "\n".join(lines)

    # ── State / City ─────────────────────────────────────────────────────────
    if "top state" in q or "best state" in q:
        top = df.groupby("State")["Amount"].sum().idxmax()
        val = df.groupby("State")["Amount"].sum().max()
        return f"🗺️ Top state by sales is **{top}** with ₹{val:,.2f}."

    if "state" in q:
        st_s = df.groupby("State")["Amount"].sum().sort_values(ascending=False).head(10)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in st_s.items()]
        return "🗺️ Top 10 States by Sales:\n" + "\n".join(lines)

    if "top city" in q or "best city" in q:
        top = df.groupby("City")["Amount"].sum().idxmax()
        val = df.groupby("City")["Amount"].sum().max()
        return f"🏙️ Top city by sales is **{top}** with ₹{val:,.2f}."

    if "city" in q:
        c_s = df.groupby("City")["Amount"].sum().sort_values(ascending=False).head(10)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in c_s.items()]
        return "🏙️ Top 10 Cities by Sales:\n" + "\n".join(lines)

    # ── Payment Mode ─────────────────────────────────────────────────────────
    if any(w in q for w in ["payment","upi","cod","card","emi"]):
        pm = df.groupby("PaymentMode")["Amount"].sum().sort_values(ascending=False)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in pm.items()]
        return "💳 Sales by Payment Mode:\n" + "\n".join(lines)

    # ── Monthly Trend ─────────────────────────────────────────────────────────
    if any(w in q for w in ["monthly","trend","month"]):
        m = df.groupby("MonthYear")["Amount"].sum().sort_index()
        best  = m.idxmax()
        worst = m.idxmin()
        return (
            f"📅 Monthly sales range from ₹{m.min():,.2f} to ₹{m.max():,.2f}.\n"
            f"Best month: **{best}** | Worst month: **{worst}**."
        )

    # ── Loss / Underperforming ────────────────────────────────────────────────
    if any(w in q for w in ["loss","underperform","negative","worst"]):
        sub_p = df.groupby("Sub-Category")["Profit"].sum().sort_values().head(5)
        lines = [f"- **{k}**: ₹{v:,.2f}" for k, v in sub_p.items()]
        return "⚠️ Worst-performing Sub-Categories by Profit:\n" + "\n".join(lines)

    # ── Forecast ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["forecast","predict","future","next month"]):
        _, fc, r2 = forecast_sales(df, months_ahead=3)
        lines = [f"- **{row['MonthYear']}**: ₹{row['Amount']:,.2f}" for _, row in fc.iterrows()]
        return (
            f"🔮 3-Month Sales Forecast (Linear Regression, R²={r2:.2f}):\n"
            + "\n".join(lines)
        )

    # ── Recommendations ───────────────────────────────────────────────────────
    if any(w in q for w in ["recommend","suggest","improve","strategy"]):
        _, recs, _ = generate_insights(df)
        return "📋 Business Recommendations:\n" + "\n".join(recs)

    # ── Fallback ─────────────────────────────────────────────────────────────
    return (
        "🤖 I can answer questions like:\n"
        "- *Total sales / profit / orders*\n"
        "- *Best/top category, state, city*\n"
        "- *Payment mode breakdown*\n"
        "- *Monthly trend*\n"
        "- *Loss-making or worst sub-categories*\n"
        "- *Sales forecast*\n"
        "- *Recommendations*\n\n"
        "Try rephrasing your question using those keywords."
    )


# ─────────────────────────────────────────────────────────────────────────────
# DARK CHART THEME  (applied to every Plotly figure)
# ─────────────────────────────────────────────────────────────────────────────
_CHART_BG   = "#0f1117"   # matches .stApp background
_CHART_SURF = "#1a1d26"   # slightly lighter surface
_CHART_TEXT = "#e8eaf0"   # primary text on dark
_CHART_GRID = "#2d3143"   # subtle gridlines

_DARK_LAYOUT = dict(
    paper_bgcolor = _CHART_BG,
    plot_bgcolor  = _CHART_SURF,
    font          = dict(color=_CHART_TEXT, size=12),
    title_font    = dict(color=_CHART_TEXT, size=14),
    xaxis         = dict(
        color          = _CHART_TEXT,
        tickfont       = dict(color=_CHART_TEXT),
        title_font     = dict(color=_CHART_TEXT),
        gridcolor      = _CHART_GRID,
        linecolor      = _CHART_GRID,
        zerolinecolor  = _CHART_GRID,
    ),
    yaxis         = dict(
        color          = _CHART_TEXT,
        tickfont       = dict(color=_CHART_TEXT),
        title_font     = dict(color=_CHART_TEXT),
        gridcolor      = _CHART_GRID,
        linecolor      = _CHART_GRID,
        zerolinecolor  = _CHART_GRID,
    ),
    legend        = dict(
        bgcolor    = _CHART_SURF,
        bordercolor= _CHART_GRID,
        font       = dict(color=_CHART_TEXT),
    ),
    coloraxis_colorbar = dict(
        tickfont  = dict(color=_CHART_TEXT),
        title_font= dict(color=_CHART_TEXT),
    ),
)


def _dark(extra: dict = None) -> dict:
    """Return _DARK_LAYOUT merged with any extra kwargs."""
    merged = dict(_DARK_LAYOUT)
    if extra:
        for k, v in extra.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ────────────────────────────────────────────────────────────
    df_full = load_data()
    validate_data(df_full)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='font-size:22px;font-weight:800;color:#f0f4ff;margin-bottom:2px;'>🛒 EcomInsight AI</div>"
            "<div style='font-size:12px;color:#9ca3af;margin-bottom:8px;'>AI-Powered E-Commerce Analytics</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.subheader("🔍 Filters")

        all_states    = sorted(df_full["State"].dropna().unique().tolist())
        sel_states    = st.multiselect("State", all_states, default=all_states, key="states")

        all_cats      = sorted(df_full["Category"].dropna().unique().tolist())
        sel_cats      = st.multiselect("Category", all_cats, default=all_cats, key="cats")

        all_pm        = sorted(df_full["PaymentMode"].dropna().unique().tolist())
        sel_pm        = st.multiselect("Payment Mode", all_pm, default=all_pm, key="pm")

        # Date range
        min_date = df_full["Order Date"].min().date()
        max_date = df_full["Order Date"].max().date()
        date_range = st.date_input(
            "Order Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        st.caption("Dataset: [Madhav E-Commerce Sales](https://www.kaggle.com/datasets/amitkumar209/madhav-e-commerce-sales-dataset)")

    # ── Apply Filters ─────────────────────────────────────────────────────────
    df = df_full.copy()
    if sel_states:
        df = df[df["State"].isin(sel_states)]
    if sel_cats:
        df = df[df["Category"].isin(sel_cats)]
    if sel_pm:
        df = df[df["PaymentMode"].isin(sel_pm)]
    if len(date_range) == 2:
        start_d, end_d = date_range
        df = df[(df["Order Date"].dt.date >= start_d) & (df["Order Date"].dt.date <= end_d)]

    if df.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar filters.")
        st.stop()

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:2px;color:#f0f4ff;'>🛒 EcomInsight AI</h1>"
        "<p style='font-size:16px;color:#c9d8f5;margin-top:0;margin-bottom:4px;font-weight:600;'>"
        "E-Commerce Analytics Dashboard</p>"
        "<p style='font-size:13px;color:#9ca3af;margin-top:0;'>"
        "AI-Powered E-Commerce Analytics &nbsp;|&nbsp; "
        "Built with IBM Bob &nbsp;·&nbsp; Author: Pranav Kurupath</p>",
        unsafe_allow_html=True,
    )

    # ── KPI ROW ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    # Calculations — all from filtered df, no hardcoding
    total_sales   = df["Amount"].sum()
    total_profit  = df["Profit"].sum()
    total_orders  = df["Order ID"].nunique()
    total_qty     = int(df["Quantity"].sum())
    avg_ov        = df.groupby("Order ID")["Amount"].sum().mean()
    margin        = (total_profit / total_sales * 100) if total_sales else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi("Total Sales",     f"₹{total_sales:,.0f}")
    with c2: kpi("Total Profit",    f"₹{total_profit:,.0f}")
    with c3: kpi("Profit Margin",   f"{margin:.1f}%")
    with c4: kpi("Total Orders",    f"{total_orders:,}")
    with c5: kpi("Avg Order Value", f"₹{avg_ov:,.0f}")
    with c6: kpi("Units Sold",      f"{total_qty:,}")

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Sales Trends",
        "🗂️ Category Analysis",
        "🗺️ Geo Analysis",
        "💳 Payment Analysis",
        "🔮 Forecasting",
        "🤖 AI Analyst",
        "💡 Insights & Recs",
    ])

    # ── TAB 1: Sales Trends ───────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-title">Monthly Sales & Profit Trends</div>', unsafe_allow_html=True)

        monthly = (
            df.groupby("MonthYear")
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"))
            .reset_index()
            .sort_values("MonthYear")
        )

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=monthly["MonthYear"], y=monthly["Sales"],
                                  mode="lines+markers", name="Sales",
                                  line=dict(color="#3b82d4", width=2)))
        fig1.add_trace(go.Scatter(x=monthly["MonthYear"], y=monthly["Profit"],
                                  mode="lines+markers", name="Profit",
                                  line=dict(color="#22c55e", width=2)))
        fig1.update_layout(**_dark({
            "title": "Monthly Sales & Profit",
            "xaxis_title": "Month",
            "yaxis_title": "Amount (₹)",
            "legend": dict(orientation="h", font=dict(color=_CHART_TEXT),
                           bgcolor=_CHART_SURF, bordercolor=_CHART_GRID),
            "height": 380,
        }))
        st.plotly_chart(fig1, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Quarterly Sales</div>', unsafe_allow_html=True)
            qtr = df.groupby("Quarter")["Amount"].sum().reset_index().sort_values("Quarter")
            fig_q = px.bar(qtr, x="Quarter", y="Amount", color="Quarter",
                           labels={"Amount":"Sales (₹)"}, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_q.update_layout(**_dark({"showlegend": False, "height": 300}))
            st.plotly_chart(fig_q, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Monthly Profit Margin (%)</div>', unsafe_allow_html=True)
            monthly["Margin"] = (monthly["Profit"] / monthly["Sales"] * 100).fillna(0)
            fig_m = px.line(monthly, x="MonthYear", y="Margin",
                            markers=True, labels={"Margin":"Profit Margin (%)"},
                            color_discrete_sequence=["#f97316"])
            fig_m.update_layout(**_dark({"height": 300}))
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown('<div class="section-title">Top 10 Customers by Sales</div>', unsafe_allow_html=True)
        top_cust = (
            df.groupby("CustomerName")["Amount"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig_cust = px.bar(top_cust, x="Amount", y="CustomerName", orientation="h",
                          color="Amount", color_continuous_scale="Blues",
                          labels={"Amount":"Sales (₹)","CustomerName":"Customer"})
        fig_cust.update_layout(**_dark({
            "height": 350,
            "yaxis": dict(autorange="reversed", color=_CHART_TEXT,
                          tickfont=dict(color=_CHART_TEXT),
                          title_font=dict(color=_CHART_TEXT),
                          gridcolor=_CHART_GRID, linecolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_cust, use_container_width=True)

    # ── TAB 2: Category Analysis ──────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-title">Category Performance</div>', unsafe_allow_html=True)

        cat_agg = (
            df.groupby("Category")
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"), Quantity=("Quantity","sum"))
            .reset_index()
        )
        cat_agg["Margin%"] = (cat_agg["Profit"] / cat_agg["Sales"] * 100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(cat_agg, values="Sales", names="Category",
                             title="Sales Share by Category",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(**_dark({"height": 340}))
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_cat = px.bar(cat_agg, x="Category", y=["Sales","Profit"], barmode="group",
                             title="Sales vs Profit by Category",
                             color_discrete_sequence=["#3b82d4","#22c55e"],
                             labels={"value":"Amount (₹)"})
            fig_cat.update_layout(**_dark({"height": 340}))
            st.plotly_chart(fig_cat, use_container_width=True)

        st.dataframe(cat_agg.style.format({"Sales":"₹{:,.0f}","Profit":"₹{:,.0f}","Margin%":"{:.1f}%"}),
                     use_container_width=True)

        st.markdown('<div class="section-title">Sub-Category Deep Dive</div>', unsafe_allow_html=True)

        sub_agg = (
            df.groupby(["Category","Sub-Category"])
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"), Quantity=("Quantity","sum"))
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        sub_agg["Margin%"] = (sub_agg["Profit"] / sub_agg["Sales"] * 100).round(1)

        fig_sub = px.bar(sub_agg.head(20), x="Sub-Category", y="Sales",
                         color="Category", title="Top 20 Sub-Categories by Sales",
                         labels={"Sales":"Sales (₹)"},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sub.update_layout(**_dark({
            "height": 380,
            "xaxis": dict(tickangle=-35, color=_CHART_TEXT,
                          tickfont=dict(color=_CHART_TEXT),
                          title_font=dict(color=_CHART_TEXT),
                          gridcolor=_CHART_GRID, linecolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_sub, use_container_width=True)

        # Profit heatmap by sub-category
        fig_sub_p = px.bar(
            sub_agg.sort_values("Profit"), x="Profit", y="Sub-Category",
            orientation="h", color="Profit",
            color_continuous_scale=["#ef4444","#f9fafb","#22c55e"],
            title="Profit by Sub-Category (Red = Loss)",
            labels={"Profit":"Profit (₹)"},
        )
        fig_sub_p.update_layout(**_dark({
            "height": 550,
            "yaxis": dict(autorange="reversed", color=_CHART_TEXT,
                          tickfont=dict(color=_CHART_TEXT),
                          title_font=dict(color=_CHART_TEXT),
                          gridcolor=_CHART_GRID, linecolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_sub_p, use_container_width=True)

    # ── TAB 3: Geo Analysis ───────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-title">State-wise Performance</div>', unsafe_allow_html=True)

        state_agg = (
            df.groupby("State")
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"), Orders=("Order ID","nunique"))
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        state_agg["Margin%"] = (state_agg["Profit"] / state_agg["Sales"] * 100).round(1)

        fig_s = px.bar(state_agg, x="State", y="Sales", color="Profit",
                       color_continuous_scale=["#ef4444","#f9fafb","#22c55e"],
                       title="Sales by State (Colour = Profit)",
                       labels={"Sales":"Sales (₹)"})
        fig_s.update_layout(**_dark({
            "height": 420,
            "xaxis": dict(tickangle=-40, color=_CHART_TEXT,
                          tickfont=dict(color=_CHART_TEXT),
                          title_font=dict(color=_CHART_TEXT),
                          gridcolor=_CHART_GRID, linecolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_s, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_sp = px.pie(state_agg.head(8), values="Sales", names="State",
                            title="Top 8 States — Sales Share",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_sp.update_layout(**_dark({"height": 340}))
            st.plotly_chart(fig_sp, use_container_width=True)
        with col2:
            st.dataframe(
                state_agg[["State","Sales","Profit","Margin%","Orders"]]
                .style.format({"Sales":"₹{:,.0f}","Profit":"₹{:,.0f}","Margin%":"{:.1f}%"}),
                use_container_width=True, height=340,
            )

        st.markdown('<div class="section-title">Top 15 Cities by Sales</div>', unsafe_allow_html=True)
        city_agg = (
            df.groupby("City")
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"))
            .reset_index()
            .sort_values("Sales", ascending=False)
            .head(15)
        )
        fig_c = px.bar(city_agg, x="City", y="Sales", color="Profit",
                       color_continuous_scale=["#ef4444","#f9fafb","#22c55e"],
                       labels={"Sales":"Sales (₹)"})
        fig_c.update_layout(**_dark({
            "height": 360,
            "xaxis": dict(tickangle=-35, color=_CHART_TEXT,
                          tickfont=dict(color=_CHART_TEXT),
                          title_font=dict(color=_CHART_TEXT),
                          gridcolor=_CHART_GRID, linecolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_c, use_container_width=True)

    # ── TAB 4: Payment Analysis ───────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-title">Payment Mode Analysis</div>', unsafe_allow_html=True)

        pm_agg = (
            df.groupby("PaymentMode")
            .agg(Sales=("Amount","sum"), Profit=("Profit","sum"),
                 Orders=("Order ID","count"), Quantity=("Quantity","sum"))
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        pm_agg["Margin%"] = (pm_agg["Profit"] / pm_agg["Sales"] * 100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig_pm_pie = px.pie(pm_agg, values="Sales", names="PaymentMode",
                                title="Sales Share by Payment Mode",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pm_pie.update_layout(**_dark({"height": 340}))
            st.plotly_chart(fig_pm_pie, use_container_width=True)
        with col2:
            fig_pm_bar = px.bar(pm_agg, x="PaymentMode", y=["Sales","Profit"],
                                barmode="group", title="Sales vs Profit by Payment Mode",
                                color_discrete_sequence=["#3b82d4","#22c55e"],
                                labels={"value":"Amount (₹)"})
            fig_pm_bar.update_layout(**_dark({"height": 340}))
            st.plotly_chart(fig_pm_bar, use_container_width=True)

        st.markdown('<div class="section-title">Payment Mode × Category Heatmap</div>', unsafe_allow_html=True)
        pm_cat = df.pivot_table(index="Category", columns="PaymentMode",
                                values="Amount", aggfunc="sum", fill_value=0)
        fig_heat = px.imshow(pm_cat, text_auto=".0f", color_continuous_scale="Blues",
                             title="Sales (₹) — Category × Payment Mode",
                             aspect="auto")
        fig_heat.update_layout(**_dark({"height": 300}))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.dataframe(
            pm_agg.style.format({"Sales":"₹{:,.0f}","Profit":"₹{:,.0f}","Margin%":"{:.1f}%"}),
            use_container_width=True,
        )

    # ── TAB 5: Forecasting ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-title">Sales Forecasting (Linear Regression)</div>', unsafe_allow_html=True)

        months_ahead = st.slider("Months to Forecast", min_value=1, max_value=6, value=3)
        combined, fc_df, r2 = forecast_sales(df, months_ahead=months_ahead)

        fig_fc = px.line(combined, x="MonthYear", y="Amount", color="Type",
                         markers=True,
                         color_discrete_map={"Historical":"#3b82d4","Forecast":"#f97316"},
                         labels={"Amount":"Sales (₹)","MonthYear":"Month"},
                         title=f"Sales Forecast — Next {months_ahead} Month(s)")
        fig_fc.update_layout(**_dark({
            "height": 420,
            "legend": dict(orientation="h", font=dict(color=_CHART_TEXT),
                           bgcolor=_CHART_SURF, bordercolor=_CHART_GRID),
        }))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.caption(f"Model: Linear Regression | R² = {r2:.3f}")

        st.markdown("**Forecasted Values:**")
        fc_display = fc_df[["MonthYear","Amount"]].copy()
        fc_display.columns = ["Month", "Forecasted Sales (₹)"]
        fc_display["Forecasted Sales (₹)"] = fc_display["Forecasted Sales (₹)"].map("₹{:,.2f}".format)
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            "**Note:** Forecasts are based on a linear regression fitted to monthly aggregated sales. "
            "The model captures the overall trend; actual results may vary due to seasonality "
            "and external factors not captured in the dataset."
        )

    # ── TAB 6: AI Analyst ─────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-title">🤖 AI Business Analyst</div>', unsafe_allow_html=True)
        st.markdown(
            "Ask a business question about the dataset. The AI Analyst will respond "
            "using the actual data loaded from your dataset."
        )

        example_queries = [
            "What is the total sales revenue?",
            "Which category has the highest profit?",
            "Top state by sales?",
            "Payment mode breakdown",
            "Monthly trend",
            "Sales forecast for next 3 months",
            "What are the loss-making sub-categories?",
            "Give me business recommendations",
        ]
        selected_eg = st.selectbox("Quick Question", ["— Select —"] + example_queries)

        user_q = st.text_input(
            "Or type your question:",
            value=selected_eg if selected_eg != "— Select —" else "",
            placeholder="e.g. What is the total profit?",
        )
        if st.button("Ask AI Analyst", type="primary") and user_q.strip():
            with st.spinner("Analysing…"):
                answer = answer_query(df, user_q)
            st.markdown("**Answer:**")
            st.markdown(answer)

    # ── TAB 7: Insights & Recommendations ────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-title">AI-Generated Insights</div>', unsafe_allow_html=True)
        insights, recs, warns = generate_insights(df)

        for ins in insights:
            st.markdown(f'<div class="insight-box">🔍 {ins}</div>', unsafe_allow_html=True)

        if warns:
            st.markdown('<div class="section-title">⚠️ Warnings / Risk Areas</div>', unsafe_allow_html=True)
            for w in warns:
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Business Recommendations</div>', unsafe_allow_html=True)
        for rec in recs:
            st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)

        st.markdown("---")
        # Quick summary table
        st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)
        summary = {
            "Total Rows (Detail Records)": len(df),
            "Unique Orders":               df["Order ID"].nunique(),
            "Unique Customers":            df["CustomerName"].nunique(),
            "Unique States":               df["State"].nunique(),
            "Unique Cities":               df["City"].nunique(),
            "Categories":                  df["Category"].nunique(),
            "Sub-Categories":              df["Sub-Category"].nunique(),
            "Payment Modes":               df["PaymentMode"].nunique(),
            "Date Range":                  f"{df['Order Date'].min().date()} → {df['Order Date'].max().date()}",
        }
        summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.table(summary_df)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
