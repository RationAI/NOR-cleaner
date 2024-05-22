import re
import sys
from datetime import datetime
from functools import partial
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from data_preparation.column_names import RECORD_COUNT_NAME
from lib.merge_records import augment_merged_x_y_df

sns.set_theme()


# Model evaluation
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def print_accuracy(y_true, y_pred, dataset: str):
    print(
        f"Accuracy for {dataset} Dataset:",
        round(
            accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            * 100,
            4,
        ),
        "%",
    )


def plot_confussion_matrix(
    y_true,
    y_pred,
    ax=None,
) -> pd.DataFrame:
    if ax is None:
        _, ax = plt.subplots()

    df_mat = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["False (Actual)", "True (Actual)"],
        columns=["False (Predicted)", "True (Predicted)"],
    )

    # Divide by the size of the dataset
    n = len(y_true)

    # Heatmap
    sns.heatmap(df_mat / n * 100, annot=False, ax=ax, cmap="Blues", fmt=".2f")

    # Set labels for annotations
    for i in range(len(df_mat)):
        for j in range(len(df_mat.columns)):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{df_mat.iloc[i, j]/n*100:.2f}%",
                ha="center",
                va="center",
                color="white" if df_mat.iloc[i, j] / n > 0.5 else "black",
                fontsize=12,
            )

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")

    return df_mat


def evaluate_by_report_count(
    data: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    record_count: tuple[int, int],
    record_count_getter: Callable[[pd.DataFrame], pd.Series],
    y_score: pd.Series | None = None,
) -> None:
    """
    Returns the accuracy for the given report count
    """
    assert y_true.shape == y_pred.shape
    assert data.index.equals(y_true.index)

    score_names = [
        "accuracy",
        "f1",
        "precision",
        "recall",
    ]

    proba_score_names = [
        "roc_auc",
    ]

    neg_label_as_pos = lambda f: partial(f, pos_label=0)
    name_to_score = {
        "accuracy": accuracy_score,
        "f1": neg_label_as_pos(f1_score),
        "roc_auc": roc_auc_score,
        "precision": neg_label_as_pos(precision_score),
        "recall": neg_label_as_pos(recall_score),
    }

    scores = {}
    for i in range(record_count[0], record_count[1] + 1):
        scores[i] = {}

        # Get the indices of the given report count
        records = record_count_getter(data)
        idx = records[records == i].index
        # Get the true and predicted values
        y_true_i = y_true.loc[idx]
        y_pred_i = y_pred.loc[idx]
        if y_score is not None:
            y_score_i = y_score.loc[idx]

        for name in score_names:
            scorer = name_to_score[name]
            scores[i]["test_" + name] = scorer(y_true_i, y_pred_i)

        if y_score is not None:
            for name in proba_score_names:
                scorer = name_to_score[name]
                scores[i]["test_" + name] = scorer(y_true_i, y_score_i)

    return scores


def get_ids_preds(
    patient_report_id: pd.DataFrame,
    id_name: str,
    pred_indices: pd.Index,
    y_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get dataframe with ids and predictions.
    """

    assert len(pred_indices.unique()) == len(
        pred_indices
    ), "Indices must be unique"

    # Get the ids for each entry
    # We are predicting the first entry of each patient
    ids = patient_report_id.loc[pred_indices, 0].reset_index(drop=True)

    # Get the labels for each entry
    labels = y_pred.reset_index(drop=True)

    assert len(ids) == len(
        labels
    ), f"Lengths of ids and labels must be equal: {len(ids)} != {len(labels)}"

    out = pd.DataFrame(pd.concat([ids, labels], axis=1))

    if len(out.columns) != 2:
        print(out.head())

    # Rename the columns
    out.columns = [id_name, "prediction"]

    return out


def get_cross_val_score(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series | None = None,
    scoring: dict[str, str] = {"accuracy": "accuracy"},
) -> dict[str, np.ndarray]:
    # Group by the PacientId column
    return cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True),
        n_jobs=8,
        scoring=scoring,
    )


def process_scoring_dict(scores: dict[str, np.array]) -> str:
    keys = sorted(scores.keys())

    out = ""
    for key in keys:
        value = scores[key]
        if len(out) > 0:
            out += "\n"

        # 95% confidence interval
        confidence_val_95 = 1.96 * value.std() / np.sqrt(len(value))
        confidence_interval = [
            value.mean() - confidence_val_95,
            value.mean() + confidence_val_95,
        ]

        dataset_name, metric_name = key.split("_", 1)
        dataset_name = dataset_name.capitalize()
        metric_name = metric_name.capitalize()

        out += f"{dataset_name: <6}: {metric_name: <15}: {value.mean():.2f}"
        out += f" ± {confidence_val_95:.3f}"
        # out += f" [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"

    return out


def process_each_records_scores(scores: dict[str, np.array]) -> str:
    out = ""
    max_val = -1
    for val in scores.keys():
        if isinstance(val, int):
            max_val = max(max_val, val)

    for i in range(2, max_val + 1):
        out += f"Group {i}\n"
        out += f"{process_scoring_dict(scores[i])}\n"
        out += "---\n"

    return out


def cross_val_str(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series | None = None,
    scoring: dict[str, str] = {"accuracy": "accuracy"},
) -> str:
    scores = get_cross_val_score(model, X, y, groups=groups, scoring=scoring)

    return process_scoring_dict(scores)


def _add_scores(
    scores,
    score_names,
    proba_score_names,
    name_to_score,
    y_train,
    y_test,
    train_preds,
    preds,
    preds_proba,
    train_preds_proba,
):
    for name in score_names:
        scorer = name_to_score[name]
        scores["train_" + name].append(scorer(y_train, train_preds))
        scores["test_" + name].append(scorer(y_test, preds))

    for name in proba_score_names:
        scorer = name_to_score[name]
        scores["train_" + name].append(scorer(y_train, train_preds_proba))
        scores["test_" + name].append(scorer(y_test, preds_proba))

    return scores


def _init_scores_dict(score_names):
    scores = {}
    # Add both test and train scores
    for score_name in score_names:
        scores["train_" + score_name] = []
        scores["test_" + score_name] = []
    return scores


def get_which_scores(
    scores: dict[str, np.ndarray], which: Literal["train", "test"]
) -> dict[str, np.ndarray]:
    """
    Get the scores for the given dataset `which`.

    Parameters:
        scores: dict[str, np.ndarray]
            Dictionary containing the scores for each metric.

        which: str
            Which scores to get. Either "train" or "test".
    """
    return {
        k.split("_", 1)[1].capitalize(): v
        for k, v in scores.items()
        if which in k
    }


def cross_validation_merged_df(
    model: Any,
    n: int,
    X_merged: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scaler: Any = None,
    n_splits: int = 5,
    each_record_eval: bool = False,
    record_count_getter: Callable[[pd.DataFrame], pd.Series] | None = None,
    augment_data: bool = False,
    random_state: int = 42,
    confusion_matrix_plot: bool = False,
) -> dict[str, np.ndarray]:
    """
    Cross-validation with data augmentation.
    The data is augmented by swapping the rows of a merged dataset.

    Parameters:
        model: Any
            Model to be used for prediction implementing fit and predict.
            If roc_auc is used, the model must also implement predict_proba.

        X_merged: pd.DataFrame
            DataFrame containing the merged data.

        groups: pd.Series
            Series containing the groups for the StratifiedGroupKFold.
            The groups must have the same length as the DataFrame.
        
        scaler: Any
            Scaler to be used for the data.

        n_splits: int
            Number of splits for the cross-validation.

        each_record_eval: bool
            Whether to evaluate the model for each record count.

        record_count_getter: Callable[[pd.DataFrame], pd.Series]
            Function to get the record count from the DataFrame.

        augment_data: bool
            Whether to augment the data by swapping the rows.

        random_state: int
            Random state for the StratifiedGroupKFold.

        confusion_matrix_plot: bool
            Whether to plot the confusion matrix.

    Returns:
        dict[str, np.ndarray]
            Dictionary containing the scores for each metric.
            The key is the metric name and the value is a numpy array
            containing the scores for each fold.
    """
    assert X_merged.index.equals(y.index)
    assert n >= 2

    score_names = [
        "accuracy",
        "f1",
        "precision",
        "recall",
    ]

    proba_score_names = [
        "roc_auc",
        # "log_loss",
    ]

    scores = _init_scores_dict(score_names + proba_score_names)

    neg_label_as_pos = lambda f: partial(f, pos_label=0)
    name_to_score = {
        "accuracy": accuracy_score,
        "f1": neg_label_as_pos(f1_score),
        "roc_auc": roc_auc_score,
        "precision": neg_label_as_pos(precision_score),
        "recall": neg_label_as_pos(recall_score),
        # "log_loss": log_loss,
    }

    if each_record_eval:
        each_record_score = {}
        for i in range(2, n + 1):
            each_record_score[i] = _init_scores_dict(
                score_names + proba_score_names
            )

    if confusion_matrix_plot:
        y_preds = []
        y_trues = []

    skf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for train_index, test_index in skf.split(X_merged, y, groups=groups):
        # Split the data into train and test folds
        X_train = X_merged.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X_merged.iloc[test_index]
        y_test = y.iloc[test_index]

        # print(X_test["RecordCount"].value_counts())
        # print(y_test.value_counts())

        # Add permutations to the train fold -- data augmentation
        if augment_data:
            X_train, y_train = augment_merged_x_y_df(X_train, y_train, n)

        assert X_train.columns.equals(X_test.columns)

        # Save before scaling
        if each_record_eval:
            record_counts_train = record_count_getter(X_train)
            record_counts_test = record_count_getter(X_test)

        # Scale data if a scaler is provided
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit the model on the train fold
        model.fit(X_train, y_train)

        # Predict on the test fold
        preds = model.predict(X_test)
        train_preds = model.predict(X_train)

        if proba_score_names:
            train_preds_proba = model.predict_proba(X_train)[:, 1]
            preds_proba = model.predict_proba(X_test)[:, 1]

        _add_scores(
            scores,
            score_names,
            proba_score_names,
            name_to_score,
            y_train,
            y_test,
            train_preds,
            preds,
            preds_proba,
            train_preds_proba,
        )

        if confusion_matrix_plot:
            y_preds.append(pd.Series(preds))
            y_trues.append(y_test.copy())

        if each_record_eval:
            # At least 2 records
            for i in range(2, n + 1):
                train_mask = record_counts_train == i
                test_mask = record_counts_test == i
                y_train_i_records = y_train[train_mask]
                y_test_i_records = y_test[test_mask]

                train_preds_i_records = pd.Series(
                    train_preds, index=y_train.index
                )[train_mask]

                preds_i_records = pd.Series(preds, index=y_test.index)[
                    test_mask
                ]

                if proba_score_names:
                    train_preds_proba_i_records = pd.Series(
                        train_preds_proba, index=y_train.index
                    )[train_mask]
                    preds_proba_i_records = pd.Series(
                        preds_proba, index=y_test.index
                    )[test_mask]

                _add_scores(
                    each_record_score[i],
                    score_names,
                    proba_score_names,
                    name_to_score,
                    y_train_i_records,
                    y_test_i_records,
                    train_preds_i_records,
                    preds_i_records,
                    preds_proba_i_records,
                    train_preds_proba_i_records,
                )

    scores = {k: np.array(v) for k, v in scores.items()}

    if confusion_matrix_plot:
        y_true = pd.concat(y_trues)
        y_pred = pd.concat(y_preds)

        _, ax = plt.subplots()
        plot_confussion_matrix(y_true, y_pred, ax=ax)
        ax.set_title("Cross Validation Confusion Matrix")

    if each_record_eval:
        each_record_score = {
            k: {k2: np.array(v2) for k2, v2 in v.items()}
            for k, v in each_record_score.items()
        }

        return scores, each_record_score

    return scores


def cross_val_non_merged_df(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n: int = 4,
    n_splits: int = 5,
    scaler: Any = None,
    random_state: int = 42,
    each_record_eval: bool = False,
    confusion_matrix_plot: bool = False,
) -> dict[str, np.ndarray]:
    """
    Cross-validation for non-merged data.

    Parameters:
        model: Any
            Model to be used for prediction implementing fit and predict.
            If roc_auc is used, the model must also implement predict_proba.

        X: pd.DataFrame
            DataFrame containing the data.

        y: pd.Series
            Series containing the labels.

        groups: pd.Series
            Series containing the groups for the StratifiedGroupKFold.
            The groups must have the same length as the DataFrame.
        
        n: int
            Number of records in a data point.
        
        n_splits: int
            Number of splits for the cross-validation.

        scaler: Any
            Scaler to be used for the data.

        random_state: int
            Random state for the StratifiedGroupKFold.
        
        each_record_eval: bool
            Whether to evaluate the model for each record count.

        confusion_matrix_plot: bool
            Whether to plot the confusion matrix.
    
    Returns:
        dict[str, np.ndarray]
            Dictionary containing the scores for each metric.
            The key is the metric name and the value is a numpy array
            containing the scores for each fold.
    """
    assert X.index.equals(y.index)

    return cross_validation_merged_df(
        model,
        n,
        X,
        y,
        groups=groups,
        n_splits=n_splits,
        each_record_eval=each_record_eval,
        random_state=random_state,
        augment_data=False,
        record_count_getter=lambda X: X["RecordCount"],
        confusion_matrix_plot=confusion_matrix_plot,
        scaler=scaler,
    )


def evaluate_model(
    model,
    names_datasets_trues: list[tuple[str, pd.DataFrame, np.ndarray]],
    groups: pd.Series | None = None,
    model_name: str | None = None,
    filename: str | None = None,
    notes: str | None = None,
    use_cross_val: bool = False,
) -> None:
    """
    Evaluate the model on the given datasets.

    Parameters:
    model: fitted model to evaluate
    groups: groups for cross-validation. If None, use the indices of the datasets.
    names_datasets_trues: list of tuples (name, dataset, true values)
    model_name: name of the model. If None, use the class name of the model.
    filename: filename to save the evaluation to. If None, the eval is printed.
    rows_merged: whether the rows of the datasets are merged. If True,
    the padded entries are removed.
    notes: notes to add to the evaluation
    use_cross_val: whether to use cross-validation
    """

    model_name = (
        model_name if model_name is not None else model.__class__.__name__
    )

    file = open(filename, "a") if filename is not None else sys.stdout

    # print function with explicit file parameter
    print_str = lambda *args, **kwargs: print(*args, **kwargs, file=file)

    # If file is not empty, add a newline
    if filename is not None and file.tell() != 0:
        print_str("\n---\n")

    names, datasets, trues = zip(*names_datasets_trues)
    y_preds = [model.predict(dataset) for dataset in datasets]

    trues_reconstructed = trues
    preds_reconstructed = y_preds

    # Print date and time
    print_str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    print_str("Evaluations")

    if notes is not None:
        print_str("\nNotes:")
        print_str(notes)

    def print_score(score_name, scorer, add_percent=False, precision=3):
        print_str(f"\n{score_name}:")
        for name, y_true, y_pred in zip(
            names, trues_reconstructed, preds_reconstructed
        ):
            print_str(
                f"{name}: {round(scorer(y_true, y_pred), precision)}{' %' if add_percent else ''}"
            )

    print_score(
        "Accuracy",
        lambda x, y: accuracy_score(x, y) * 100,
        add_percent=True,
        precision=4,
    )

    neg_label_pos = lambda func: (
        lambda y_true, y_pred: func(y_true, y_pred, pos_label=0)
    )
    print_score("F1 Score", neg_label_pos(f1_score))
    print_score("Precision", neg_label_pos(precision_score))
    print_score("Recall", neg_label_pos(recall_score))

    print_str("\nAUC:")
    auc = lambda y_true, y_score: round(roc_auc_score(y_true, y_score), 3)
    for name, X_set, y_true, y_true_reconstructed in zip(
        names, datasets, trues, trues_reconstructed
    ):
        y_probs = pd.DataFrame(model.predict_proba(X_set)[:, 1])
        print_str(f"{name}:", auc(y_true_reconstructed, y_probs))

    print_str("\nValidation dataset confusion matrix:")
    val_idx = -1
    for i, name in enumerate(names):
        if "validation" in name.lower():
            val_idx = i

    if val_idx == -1:
        print("Validation dataset not found")
    else:
        val_conf_mat = confusion_matrix(
            trues_reconstructed[val_idx], preds_reconstructed[val_idx]
        )
        print_str("TN:", val_conf_mat[0, 0])
        print_str("FP:", val_conf_mat[0, 1])
        print_str("FN:", val_conf_mat[1, 0])
        print_str("TP:", val_conf_mat[1, 1])

    X_df = pd.concat(datasets, ignore_index=True)
    y_df = pd.concat(trues, ignore_index=True)

    if use_cross_val:
        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
        }
        # Cross-validation evaluation
        print_str("\nCross-validation:")
        print_str(
            cross_val_str(model, X_df, y_df, groups=groups, scoring=scoring)
        )

    print("\nConfusion matrices (plots):")

    _, axes = plt.subplots(1, len(names), figsize=(13.33 * len(names), 10))

    if len(names) == 1:
        # Make axes iterable
        axes = [axes]

    for name, y_true, y_pred, ax in zip(
        names, trues_reconstructed, preds_reconstructed, axes
    ):
        plot_confussion_matrix(y_true, y_pred, ax=ax)
        ax.set_title(
            f"Dataset: {name}"
            + (f"; Model: {model_name}" if model_name is not None else "")
        )

    if filename is not None:
        file.close()
