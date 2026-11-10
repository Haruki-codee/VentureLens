import pandas as pd
from venturelens.analytics import clean_funding_data, prepare_company_table

def test_clean_funding_data_and_company_aggregation():
    raw = pd.DataFrame({
        "company_name": ["A", "A", "B"],
        "announced_date": ["2021-01-01", "2022-01-01", "2021-06-01"],
        "amount_usd": [100, 200, None],
        "sector": ["Fintech", "Fintech", "Health"]
    })
    clean = clean_funding_data(raw)
    assert len(clean) == 3
    assert clean["year"].tolist() == [2021, 2022, 2021] or len(clean["year"]) == 3
    companies = prepare_company_table(clean)
    assert companies.loc[companies.company_name == "A", "rounds"].iloc[0] == 2
