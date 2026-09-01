import pytest
import pandas as pd

from ds_fraud_detection_app.components.engineer_features import engineer_all_features


@pytest.fixture
def sample_df_residence_country_names():
    df_test = pd.DataFrame(
        {
            "bank_names": [
                "wise",
                "bbva",
                "revolut",
                "hsbc",
                "nab",
            ],
            "payment_methods": ["paypal", "visa", "mc", "visa", "visa"],
            "residence_country": [
                "United Kingdom",
                "Spain",
                "France",
                "United Kingdom",
                "Australia",
            ],
            "bank_country_codes": [
                "gb",
                "es",
                "gb",
                "de",
                "au",
            ],
            "traffic_sessions_countries": [
                ["United Kingdom", "Spain"],
                ["Spain"],
                ["France"],
                ["Germany", "Australia"],
                ["Australia"],
            ],
        }
    )
    return df_test


def test_engineer_all_features(sample_df_residence_country_names):
    # Assert all new feature columns added
    n_original_columns = len(sample_df_residence_country_names.columns)
    df_result = engineer_all_features(sample_df_residence_country_names)

    # Assert correct number of added columns
    assert len(df_result.columns) == n_original_columns + 4

    # Assert all new features were added
    expected_columns = {
        "is_online_bank_used",
        "is_non_card_payment_method_used",
        "is_residence_bank_country_matched",
        "is_residence_traffic_sessions_country_matched",
    }
    assert expected_columns.issubset(set(df_result.columns))

    # Assert residence_country_code column was dropped after being used
    assert "residence_country_code" not in df_result.columns
