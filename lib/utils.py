from functools import partial
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.column_names import (
    PATIENT_ID_NAME_ENG,
    TARGET_COLUMN_ENG,
    RECORD_ID_NAME_ENG,
)
from lib.load_dataset import get_ready_data

TO_DROP_IDS = [PATIENT_ID_NAME_ENG, RECORD_ID_NAME_ENG]


Train_Dataset = pd.DataFrame
Test_Dataset = pd.DataFrame
Val_Dataset = pd.DataFrame
Train_Predictions = pd.DataFrame
Test_Predictions = pd.DataFrame
Val_Predictions = pd.DataFrame


def split_by_ids(
    X: pd.DataFrame,
    y: pd.DataFrame,
    ids: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    verbose: bool = False,
    seed: int = 42,
) -> tuple[
    Train_Dataset,
    Train_Predictions,
    Val_Dataset,
    Val_Predictions,
    Test_Dataset,
    Test_Predictions,
]:
    assert test_size + val_size <= 1

    unique_ids = ids.unique()

    # Split the unique IDs into train and test sets
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=seed
    )

    # Update val_size to be a fraction of the remaining data
    val_size = val_size / (1 - test_size)

    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_size, random_state=seed
    )

    # Assert that all IDs are unique between the sets
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(test_ids) & set(val_ids)) == 0
    assert len(set(train_ids) | set(test_ids) | set(val_ids)) == len(
        unique_ids
    )

    # Use the IDs to filter the original dataset
    train_mask = ids.isin(train_ids)
    X_train = X[train_mask]
    y_train = y[train_mask]

    test_mask = ids.isin(test_ids)
    X_test = X[test_mask]
    y_test = y[test_mask]

    val_mask = ids.isin(val_ids)
    X_val = X[val_mask]
    y_val = y[val_mask]

    if verbose:
        print("Train shape:", X_train.shape)
        print("Validation shape:", X_val.shape)
        print("Test shape:", X_test.shape)
        print("--------------------")
        print("Number of unique ids:")
        print("Train:", len(train_ids))
        print("Validation:", len(val_ids))
        print("Test:", len(test_ids))

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def df_drop_ids(df: pd.DataFrame, to_drop: list[str]):
    for col in to_drop:
        if col not in df.columns:
            continue

        df = df.drop(col, axis=1)

    return df


def drop_ids_X_datasets(X_train, X_test, X_val, no_drop: list[str] = []):
    for col in TO_DROP_IDS:
        if col not in X_train.columns or col in no_drop:
            continue

        # Drop the column from X_train, X_test and X_val
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
        X_val = X_val.drop(col, axis=1)

    return X_train, X_test, X_val


def get_X_y(
    getter: Callable[[], pd.DataFrame] = partial(
        get_ready_data, which="2019-2021"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the X and y datasets from the ready data.

    Parameters:
        getter: Callable[[], pd.DataFrame]
            Function to get the dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Tuple of the X and y datasets, respectively.
    """
    data_ready = getter()

    X = data_ready.drop(columns=[TARGET_COLUMN_ENG])
    y = data_ready[TARGET_COLUMN_ENG]

    return X, y


def get_split_data(
    X: pd.DataFrame | None = None,
    y: pd.DataFrame | None = None,
    verbose: bool = False,
    drop_ids: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Get the split data for the training, validation and testing.

    Parameters:
        X: pd.DataFrame | None
            The features dataset. If None, it will be loaded by function `get_X_y`.
        y: pd.DataFrame | None
            The target dataset. If None, it will be loaded by function `get_X_y`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Tuple of the X_train, y_train, X_val, y_val, X_test, y_test datasets.
    """
    if X is None or y is None:
        X, y = get_X_y()

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_ids(
        X,
        y,
        ids=X[PATIENT_ID_NAME],
        verbose=verbose,
    )

    if drop_ids:
        X_train, X_test, X_val = drop_ids_X_datasets(X_train, X_test, X_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


def drop_ids(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=TO_DROP_IDS)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_split_data(
        drop_ids=True
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
