import pandas as pd
from typing import Optional
from sklearn.compose import ColumnTransformer

from ds_package_classifier.ml_classifier import MLClassifier, ModelProtocol


class FraudClassifier(MLClassifier):

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: ModelProtocol,
        medium_warning_level_threshold: float = 0.5,
        column_transformer: Optional[ColumnTransformer] = None,
        preprocessing_step: str = "preprocessing_step",
        classifier_step: str = "classifier_step",
    ):
        super().__init__(
            df,
            target_column,
            model,
            column_transformer=column_transformer,
            preprocessing_step=preprocessing_step,
            classifier_step=classifier_step,
        )

        # Variable to be stored with classifier in GCP that will be needed for predictions
        self.top_binarized_multi_category_features = None
        self.medium_warning_level_threshold = medium_warning_level_threshold
        self.high_warning_level_threshold = None
