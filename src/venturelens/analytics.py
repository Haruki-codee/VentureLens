import pandas as pd
import numpy as np

ALIASES = {
    "announced_date": ["announced_date", "date", "funding_date"],
    "company_name": ["company_name", "company", "startup", "name"],
    "amount_usd": ["amount_usd", "amount", "funding_amount_usd"],
    "sector": ["sector", "industry"],
    "city": ["city", "location"],
    "stage": ["stage", "funding_stage", "round"],
    "lead_investor": ["lead_investor", "investor", "lead"],
    "investor_count": ["investor_count", "num_investors"],
    "founded_year": ["founded_year", "year_founded"],
}

def _rename_known_columns(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    renames = {}
    for target, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in lower:
                renames[lower[candidate]] = target
                break
    return df.rename(columns=renames)

def clean_funding_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = _rename_known_columns(raw.copy())
    required = ["company_name", "announced_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {list(df.columns)}")
    for col in ["amount_usd", "investor_count", "founded_year"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sector", "city", "stage", "lead_investor"]:
        if col not in df:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["company_name"] = df["company_name"].fillna("Unknown").astype(str).str.strip()
    df["announced_date"] = pd.to_datetime(df["announced_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["announced_date"])
    if "year" not in df:
        df["year"] = df["announced_date"].dt.year
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["announced_date"].dt.year)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["company_name", "announced_date"]).reset_index(drop=True)
    return df

def market_summary(df):
    return {
        "rounds": int(len(df)),
        "companies": int(df["company_name"].nunique()),
        "disclosed_amount_usd": float(df["amount_usd"].sum()) if df["amount_usd"].notna().any() else None,
    }

def prepare_company_table(df):
    return (df.groupby("company_name", as_index=False)
        .agg(sector=("sector", "last"), city=("city", "last"),
             rounds=("company_name", "size"), total_disclosed_usd=("amount_usd", "sum"),
             last_round_date=("announced_date", "max"), latest_stage=("stage", "last"))
        .sort_values("last_round_date", ascending=False))
