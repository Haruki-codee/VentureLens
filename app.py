from pathlib import Path
import sys
import streamlit as st
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from venturelens.data import load_local_data, DATA_PATH
from venturelens.analytics import clean_funding_data, market_summary, prepare_company_table
from venturelens.modeling import load_model_artifacts, build_training_data

st.set_page_config(page_title="VentureLens | Startup Intelligence", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main .block-container {padding-top: 1.6rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: rgba(120,120,120,.07); border: 1px solid rgba(120,120,120,.15); padding: 14px; border-radius: 12px;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(115deg,#10233f,#1b4665); color:white; margin-bottom:1rem;}
    .hero h1 {color:white; margin:0;} .hero p {color:#d7e5f2; margin:.4rem 0 0 0;}
    .small-note {color: #777; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>VentureLens</h1>
<p>Indian startup funding intelligence · Market analytics · Funding-activity signals</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_data():
    raw = load_local_data()
    return clean_funding_data(raw)

df = None
try:
    df = get_data()
except Exception as exc:
    st.warning("The dataset is not available yet.")
    st.error(str(exc))
    st.markdown("""
    **Get started**
    1. Install dependencies: `pip install -r requirements.txt`
    2. Download the dataset: `python scripts/download_data.py`
    3. Start the dashboard: `streamlit run app.py`

    The downloader retrieves the public Fliar BI Indian Startup Funding dataset from Hugging Face.
    """)
    st.stop()

if df.empty:
    st.error("The dataset loaded, but no usable funding records were found.")
    st.stop()

summary = market_summary(df)
companies = prepare_company_table(df)

with st.sidebar:
    st.header("Explore VentureLens")
    page = st.radio("Workspace", [
        "Market Overview", "Startup Explorer", "Funding Signal Model",
        "Deal Flow", "Methodology & Data"
    ])
    st.divider()
    st.caption(f"{len(df):,} funding-round records loaded")
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    if years:
        selected_years = st.slider("Funding year range", min_value=int(min(years)),
                                   max_value=int(max(years)),
                                   value=(int(min(years)), int(max(years))))
    else:
        selected_years = None
    sectors = ["All"] + sorted([x for x in df["sector"].dropna().astype(str).unique() if x.strip()])
    selected_sector = st.selectbox("Sector", sectors)

filtered = df.copy()
if selected_years:
    filtered = filtered[filtered["year"].between(selected_years[0], selected_years[1])]
if selected_sector != "All":
    filtered = filtered[filtered["sector"].astype(str) == selected_sector]

if page == "Market Overview":
    st.subheader("Market Overview")
    st.caption("Funding activity in the selected period and sector. Amounts are USD as supplied by the dataset.")
    c1, c2, c3, c4 = st.columns(4)
    total_amount = filtered["amount_usd"].sum(min_count=1)
    c1.metric("Funding rounds", f"{len(filtered):,}")
    c2.metric("Companies", f"{filtered['company_name'].nunique():,}")
    c3.metric("Disclosed funding", f"${total_amount/1e9:,.2f}B" if total_amount == total_amount else "N/A")
    c4.metric("Median disclosed round", f"${filtered['amount_usd'].median()/1e6:,.1f}M" if filtered["amount_usd"].notna().any() else "N/A")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Funding rounds by year")
        yearly = filtered.dropna(subset=["year"]).groupby("year").size().rename("Rounds")
        st.bar_chart(yearly)
    with right:
        st.markdown("#### Disclosed funding by sector")
        sector_amt = filtered.dropna(subset=["amount_usd"]).groupby("sector")["amount_usd"].sum().sort_values(ascending=False).head(12)
        if not sector_amt.empty:
            st.bar_chart(sector_amt)
        else:
            st.info("No disclosed funding amounts for this selection.")

    st.markdown("#### Funding stage mix")
    stage_mix = filtered["stage"].fillna("Unknown").value_counts().head(12)
    if not stage_mix.empty:
        st.bar_chart(stage_mix)
    st.markdown("#### Recent rounds")
    cols = ["announced_date", "company_name", "sector", "city", "stage", "amount_usd", "lead_investor"]
    st.dataframe(filtered.sort_values("announced_date", ascending=False)[cols].head(25),
                 use_container_width=True, hide_index=True)

elif page == "Startup Explorer":
    st.subheader("Startup Explorer")
    search = st.text_input("Search company name")
    company_view = companies.copy()
    if search:
        company_view = company_view[company_view["company_name"].str.contains(search, case=False, na=False)]
    if selected_sector != "All":
        company_view = company_view[company_view["sector"].astype(str) == selected_sector]
    st.caption("Company-level history aggregated from the funding-round records available in this dataset.")
    st.dataframe(company_view.sort_values("last_round_date", ascending=False),
                 use_container_width=True, hide_index=True)
    if not company_view.empty:
        chosen = st.selectbox("Inspect a company", company_view["company_name"].dropna().unique())
        detail = df[df["company_name"] == chosen].sort_values("announced_date")
        st.markdown(f"#### {chosen}")
        st.line_chart(detail.set_index("announced_date")["amount_usd"])
        st.dataframe(detail[["announced_date","stage","amount_usd","lead_investor","investor_count"]],
                     use_container_width=True, hide_index=True)

elif page == "Funding Signal Model":
    st.subheader("Funding-Activity Signal")
    st.info("This model estimates whether a company has another recorded funding round within 24 months after an observed round. It is **not** a prediction of profitability, valuation, investment return, or whether a VC should invest.")
    artifacts = load_model_artifacts(ROOT / "models")
    if artifacts is None:
        st.warning("No trained model artifacts found yet.")
        st.markdown("Train the model with:")
        st.code("python scripts/train_model.py")
        st.markdown("Training uses chronological validation and excludes rows without a fully observable 24-month outcome window.")
    else:
        model, metrics, feature_columns = artifacts
        m1, m2, m3 = st.columns(3)
        m1.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics.get("roc_auc") is not None else "N/A")
        m2.metric("PR-AUC", f"{metrics.get('pr_auc', float('nan')):.3f}" if metrics.get("pr_auc") is not None else "N/A")
        m3.metric("Test records", f"{metrics.get('test_rows', 0):,}")
        st.caption(f"Model: {metrics.get('model_name', 'Unknown')} · Test period: {metrics.get('test_period', 'Not recorded')}")
        st.markdown("#### Score a funding-round profile")
        a, b, c = st.columns(3)
        sector_options = sorted(df["sector"].dropna().astype(str).unique().tolist())
        stage_options = sorted(df["stage"].dropna().astype(str).unique().tolist())
        city_options = sorted(df["city"].dropna().astype(str).unique().tolist())
        sector = a.selectbox("Sector", sector_options or ["Unknown"])
        stage = b.selectbox("Stage", stage_options or ["Unknown"])
        city = c.selectbox("City", city_options or ["Unknown"])
        d, e, f = st.columns(3)
        amount = d.number_input("Round amount (USD)", min_value=0.0, value=2_000_000.0, step=250_000.0)
        investors = e.number_input("Investor count", min_value=0, max_value=200, value=3)
        age = f.number_input("Startup age at round (years)", min_value=0.0, max_value=80.0, value=4.0)
        hist_count = st.number_input("Previous recorded rounds", min_value=0, max_value=100, value=1)
        hist_total = st.number_input("Previous disclosed funding total (USD)", min_value=0.0, value=1_000_000.0, step=250_000.0)
        days_prev = st.number_input("Days since previous round (0 if first)", min_value=0, max_value=10000, value=365)
        prev_amount = st.number_input("Previous round amount (USD)", min_value=0.0, value=1_000_000.0, step=250_000.0)
        if st.button("Estimate funding-activity probability", type="primary"):
            import pandas as pd
            row = pd.DataFrame([{
                "sector": sector, "stage": stage, "city": city, "amount_usd": amount,
                "investor_count": investors, "startup_age": age,
                "prior_round_count": hist_count, "prior_funding_total": hist_total,
                "days_since_prev_round": days_prev, "prev_amount_usd": prev_amount
            }])
            row = row.reindex(columns=feature_columns)
            prob = float(model.predict_proba(row)[0, 1])
            st.metric("Estimated probability of another recorded round within 24 months", f"{prob:.1%}")
            st.caption("A historical-pattern estimate only. Dataset coverage and the 24-month label definition limit interpretation.")

elif page == "Deal Flow":
    st.subheader("Deal Flow Screener")
    st.caption("A filterable list of recorded funding events. This is not a live deal-sourcing feed.")
    x, y = st.columns(2)
    min_amount = x.number_input("Minimum disclosed round amount (USD)", min_value=0, value=0, step=1_000_000)
    keyword = y.text_input("Investor contains")
    deals = filtered[filtered["amount_usd"].fillna(0) >= min_amount]
    if keyword:
        deals = deals[deals["lead_investor"].astype(str).str.contains(keyword, case=False, na=False)]
    st.dataframe(deals.sort_values("announced_date", ascending=False)[
        ["announced_date","company_name","sector","stage","amount_usd","lead_investor","investor_count","news_url"]
    ], use_container_width=True, hide_index=True, column_config={
        "news_url": st.column_config.LinkColumn("Source article")
    })

else:
    st.subheader("Methodology & Data")
    st.markdown("""
    **Source:** Fliar BI, *Indian Startup Funding Rounds 2015–2026*, hosted on Hugging Face:
    [fliarbi/indian-startup-funding](https://huggingface.co/datasets/fliarbi/indian-startup-funding).

    **Dataset limitations**
    - The published sample contains a subset of the full funding-round collection and is not a census of the Indian startup ecosystem.
    - News-derived coverage is stronger for larger and English-reported funding events.
    - The source reports dates through June 2026; the most recent years have shorter observation windows.
    - Funding amounts may be undisclosed or normalized to USD by the publisher. Treat amounts as estimates where indicated.
    - Sector and stage labels may be inconsistent across reports.
    - The dataset is licensed **CC BY-NC 4.0**. This project is for non-commercial educational/portfolio use unless you obtain separate permission.

    **Prediction target:** another recorded funding round within 730 days after the index round. Rows are eligible only when the dataset end date gives a complete 730-day observation window. The model uses a chronological holdout; it does not randomly split records.

    **Not a fund-performance product:** real IRR, TVPI, DPI, MOIC, and fund-level returns require fund cash flows, entry valuations, exit proceeds, and portfolio marks. Those data are not present here, so VentureLens does not fabricate fund returns.
    """)
    st.markdown("#### Available fields")
    st.write(", ".join(df.columns.astype(str).tolist()))
    st.markdown("#### Data quality snapshot")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing %": [round(df[c].isna().mean()*100, 1) for c in df.columns],
        "Unique values": [df[c].nunique(dropna=True) for c in df.columns]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

st.divider()
st.caption("VentureLens · Educational analytics prototype · Not investment advice.")
