import pandas as pd

from ds_fraud_detection_app.utils.utils import load_config
from ds_fraud_detection_app.utils import feature_engineering


def engineer_all_features(df_features: pd.DataFrame):

    config = load_config("config.yaml")
    online_banks = config["payment_methods"]["online_banks"]
    non_card_payment_methods = config["payment_methods"]["non_card_payment_methods"]

    df_features = feature_engineering.build_residence_country_code(df_features)
    df_features = feature_engineering.engineer_is_online_bank_used(
        df_features, online_banks
    )
    df_features = feature_engineering.engineer_is_non_card_payment_method_used(
        df_features, non_card_payment_methods
    )
    df_features = feature_engineering.engineer_is_residence_bank_country_matched(
        df_features
    )
    df_features = (
        feature_engineering.engineer_is_residence_traffic_sessions_country_matched(
            df_features
        )
    )
    df_features.drop(columns=["residence_country_code"], inplace=True)
    return df_features
