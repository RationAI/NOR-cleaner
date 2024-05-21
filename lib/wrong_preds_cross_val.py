import pandas as pd
import numpy as np

from typing import Any

from sklearn.model_selection import StratifiedKFold

from lib.permutations import augment_merged_x_y_df


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
            Whether to print the progress.
        add_proba: bool
            Whether to add the predicted probabilities to the DataFrame.

    Returns:
        pd.DataFrame
            DataFrame containing the indices of the wrong predictions.
    """

    if augment_data:
        if n <= 0:
            raise ValueError("n should be at least one")

    # Create empty DataFrame to store wrong predictions
    wrong_preds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(
        skf.split(X, y, groups=groups)
    ):
        # Split the data into train and test folds
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        if augment_data:
            X_train, y_train =  augment_merged_x_y_df(X_train, y_train, n)

        # Fit the model on the train fold
        model.fit(X_train, y_train)

        # Predict on the test fold
        preds = model.predict(X_test)

        wrong_preds_df = (
            X_test[y_test != preds].index.to_frame(name="index")
        )

        if add_proba:
            positive_preds = model.predict_proba(X_test)[:, 1]
            # Add the predicted probabilities to the DataFrame
            wrong_preds_df["proba"] = positive_preds[y_test != preds]

        # Save the wrong predictions
        wrong_preds.append(wrong_preds_df)

        if verbose:
            print(f"Fold: {i}")
            print(f"Accuracy: {np.mean(y_test == preds)}")
            print(f"Number of wrong predictions: {len(wrong_preds_df)}")

    # Concatenate the wrong predictions from all folds
    return pd.concat(wrong_preds).reset_index(drop=True)
