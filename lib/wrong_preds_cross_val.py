import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from data_preparation.merge_records import augment_merged_x_y_df
from scripts.constants import MERGED_DATA_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


def get_wrong_preds_cross_val(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: list | pd.Series | None = None,
    verbose: bool = False,
    add_proba: bool = False,
    augment_data: bool = False,
    n: int = -1,
) -> pd.DataFrame:
    """
    Function which finds wrong predictions of a model using cross-validation, i.e.
    divides the df into 5 folds, fits the model on 4 folds and predicts on 1 fold,
    and saves the wrong predictions of the test fold.

    Parameters:
        model: Any
            Model to be used for prediction implementing fit, predict
            and predict_proba (if add_proba is True) methods.
        X: pd.DataFrame
            DataFrame containing the features.
        y: pd.DataFrame
            DataFrame containing the target variable.
        groups: list | pd.Series | None
            List or Series containing the groups for the StratifiedKFold.
        verbose: bool
            Whether to log the progress.
        add_proba: bool
            Whether to add the predicted probabilities to the DataFrame.
        augment_data: bool
            Whether to augment the data using the `augment_merged_x_y_df` function.
            If True, n should be a positive integer.

    Returns:
        pd.DataFrame
            DataFrame containing the indices of the wrong predictions.
    """

    if augment_data and n <= 0:
        raise ValueError("n should be a positive integer.")

    # Create empty DataFrame to store wrong predictions
    wrong_preds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    total_wrong_preds = 0
    for i, (train_index, test_index) in enumerate(
        skf.split(X, y, groups=groups)
    ):
        # Split the data into train and test folds
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test: NDArray = y.iloc[test_index].to_numpy().ravel()

        if augment_data:
            X_train, y_train = augment_merged_x_y_df(X_train, y_train, n)

        # Fit the model on the train fold
        model.fit(X_train, y_train)

        # Predict on the test fold
        preds: NDArray = model.predict(X_test)

        wrong_preds_df = X_test[y_test != preds].index.to_frame(name="index")

        if add_proba:
            positive_preds = model.predict_proba(X_test)[:, 1]
            # Add the predicted probabilities to the DataFrame
            wrong_preds_df["proba"] = positive_preds[y_test != preds]

        # Save the wrong predictions
        wrong_preds.append(wrong_preds_df)

        total_wrong_preds += len(wrong_preds_df)

        if verbose:
            logger.info(f"Fold: {i}")
            logger.info(f"Accuracy: {np.mean(y_test == preds)}")
            logger.info(f"Number of wrong predictions: {len(wrong_preds_df)}")

    if verbose:
        logger.info(f"Total number of wrong predictions: {total_wrong_preds}")

    # Concatenate the wrong predictions from all folds
    return pd.concat(wrong_preds).reset_index(drop=True)


if __name__ == "__main__":
    # Load the data
    import xgboost as xgb

    from lib.load_dataset import load_merged_data_from_csv

    logging.basicConfig(level=logging.INFO)

    X, y, _, _ = load_merged_data_from_csv(MERGED_DATA_PATH)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    wrong_preds = get_wrong_preds_cross_val(
        model, X, y, verbose=True, add_proba=True
    )
