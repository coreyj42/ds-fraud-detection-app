import pandas as pd
from copy import deepcopy
import yaml
from importlib import resources
from typing import Dict


def convert_date_columns_to_datetime(
    df: pd.DataFrame,
):
    """
    Convert any date columns from the dataframe to pandas datetime

    __ PARAMS __
    :df: pd.DataFrame
        The input dataframe
    """

    # Copy
    tmp = deepcopy(df)

    # Convert columns with dbdate type to datetime columns
    for col in tmp.columns:
        if str(tmp[col].dtype) == "dbdate":
            tmp[col] = pd.to_datetime(tmp[col], errors="coerce")

    return tmp


def load_config(file_name: str) -> Dict:
    """Load a YAML config file from the installed package."""
    with resources.files("ds_fraud_detection_app").joinpath(f"config/{file_name}").open(
        "r"
    ) as f:
        return yaml.safe_load(f)
