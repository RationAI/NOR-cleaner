import itertools
from typing import Any

import pandas as pd


def merge_entries(
    df: pd.DataFrame,
    group_col: str,
    n: int,
    fillna: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Merge the entries of a dataframe.

    Parameters:
        df:
            Dataframe to merge
        n:
            Number of entries to merge. If there are less than n entries,
            other entries will be filled using `fillna`. If there are more than n entries,
            the entries will be dropped.
        fillna:
            Dictionary with values to fill the missing values with. If None, zero is used to
            fill the missing values.
    """
    # Create a copy of the dataframe
    df = df.copy()

    # Drop the entries where the number of entries is greater than n
    df = df[df.groupby(group_col)[group_col].transform("count") <= n]

    # Create a column with the number of entries
    df["entry"] = df.groupby(group_col).cumcount()

    # Pivot the dataframe
    df_pivot = df.pivot(index=group_col, columns="entry", values=df.columns)

    # Drop entry column
    df_pivot.drop(columns="entry", inplace=True)

    # Fill the missing values
    if fillna is None:
        df_pivot = df_pivot.fillna(0)
    else:
        for col, fill in fillna.items():
            if col not in df.columns:
                print(f"{col} not in columns")
                continue

            for i in range(n):
                df_pivot[(col, i)] = df_pivot[(col, i)].fillna(fill)

    # If there are null values in the df, throw an error
    if not df_pivot.notnull().all().all():
        raise ValueError(
            f"There are null values in the dataframe: {df_pivot.columns[df_pivot.isnull().any()]}"
        )

    return df_pivot


def swap_indices(indices: list, ids1: list, ids2: list) -> list:
    """
    Swap the indices of ids1 and ids2 in indices.
    Parameters:
    indices: list of indices
    ids1: list of ids to swap
    ids2: list of ids to swap
    """
    if len(ids1) != len(ids2):
        raise ValueError(f"Lengths of ids1 and ids2 must be equal: {len(ids1)} != {len(ids2)}")
    
    if (x := len(set(indices))) != len(indices):
        raise ValueError(f"Indices must be unique: {x} != {len(indices)}")

    pos1 = []
    pos2 = []

    k1 = 0
    k2 = 0
    for i, idx in enumerate(indices):
        if k1 < len(ids1) and idx == ids1[k1]:
            pos1.append(i)
            k1 += 1
        elif k2 < len(ids2) and idx == ids2[k2]:
            pos2.append(i)
            k2 += 1

    assert len(pos1) == len(
        ids1
    ), f"All ids1 must be present in indices: {len(pos1)} != {len(ids1)}"

    assert len(pos2) == len(
        ids2
    ), f"All ids2 must be present in indices: {len(pos2)} != {len(ids2)}"

    # Create a copy of the indices
    indices = indices.copy()

    for pos1, pos2 in zip(pos1, pos2):
        indices[pos1], indices[pos2] = indices[pos2], indices[pos1]

    return indices


def swap_columns(df: pd.DataFrame, col1: Any, col2: Any) -> pd.DataFrame:
    """
    Swap two columns of a dataframe. Only swap the values, not the column names.

    Parameters:
        df: pd.DataFrame
            Dataframe to swap columns in
        col1: str
            First column to swap
        col2: str
            Second column to swap

    Returns:
        pd.DataFrame
            Dataframe with swapped columns
    """
    if col1 not in df.columns:
        raise ValueError(f"{col1} not in columns")
    if col2 not in df.columns:
        raise ValueError(f"{col2} not in columns")

    # Assert that the indices are unique
    assert df.index.is_unique, "Indices must be unique"

    # Create a copy of the dataframe
    df = df.copy()

    if col1 == col2:
        return df

    # Get the indices of the columns to swap
    idx1 = df.columns.get_loc(col1)
    idx2 = df.columns.get_loc(col2)

    df.iloc[:, [idx1, idx2]] = df.iloc[:, [idx2, idx1]]

    return df


def swap_rows_merged_dataframe(
    df: pd.DataFrame, row_idx1: int, row_idx2: int
) -> pd.DataFrame:
    """
    Swap two rows of a merged dataframe. The indices of the rows are swapped.
    Parameters:
        df: pd.DataFrame
            Merged dataframe to swap rows in
        row_idx1: int
            First row to swap
        row_idx2: int
            Second row to swap

    Returns:
        pd.DataFrame
            Dataframe with swapped rows
    """
    # Create a copy of the dataframe
    df = df.copy()

    if row_idx1 == row_idx2:
        return df

    cols_to_swap = set()
    for col_name, _ in df.columns:
        if (col_name, row_idx1) in df.columns and (
            col_name,
            row_idx2,
        ) in df.columns:
            cols_to_swap.add(col_name)

    for col_name in cols_to_swap:
        df = swap_columns(df, (col_name, row_idx1), (col_name, row_idx2))

    return df


def merge_groups_each_row(
    df: pd.DataFrame,
    group_col: str,
    n: int,
    drop_padded_by: str | None = None,
    null_value: int = 0,
    fillna: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Merge entries of each group of a dataframe such that each entry has a row,
    where the entry is at the zeroth position, and the other entries of the gorup
    are merged with the zeroth entry.

    Parameters:
        df:
            dataframe to add permutations to
        group_col:
            column to group by
        n:
            number of entries per group. If less than n entries are present,
            other entries will be filled with zero if `fillna` is None, otherwise
            they will be filled with `fillna`.
        drop_padded_by:
            column to drop padded rows by. If None, padded rows are not dropped.
        null_value:
            value to drop padded rows by. Default is 0.
        fillna:
            dictionary with values to fill the missing values with. If None, zero is used to
            fill the missing values.

    The entries where first row is padded are dropped.
    """

    # Create a copy of the dataframe
    df = df.copy()

    # Assert that the number of entries is at most n
    assert (
        df.groupby(group_col)[group_col].transform("count") <= n
    ).all(), f"Number of entries is not at most {n}"

    merge_func = lambda dataframe: merge_entries(
        dataframe, group_col, n, fillna=fillna
    )

    perms = [merge_func(df)]
    merged_df = perms[0]
    for i in range(1, n):
        perms.append(swap_rows_merged_dataframe(merged_df, 0, i))

        # Drop padded rows
        if drop_padded_by is not None:
            perms[-1] = drop_padded_rows(
                perms[-1], drop_padded_by, null_value=null_value
            )

    return pd.concat(perms, axis=0, ignore_index=True)


def augment_merged_x_y_df(
    X: pd.DataFrame, y: pd.Series, n: int
) -> tuple[pd.DataFrame, pd.Series]:

    assert X.index.equals(y.index), "Indices should be equal"

    merged_df = pd.concat([X, y], axis=1)

    # Set last col name to make augmentation work properly
    column_names = list(merged_df.columns)
    column_names[-1] = ("Pred", 0)
    merged_df.columns = column_names

    merged_df = augment_merged_df(merged_df, n)

    X_out = merged_df.iloc[:, :-1].copy()
    column_names.pop()
    X_out.columns = pd.MultiIndex.from_tuples(column_names)

    y_out = merged_df.iloc[:, -1].copy()
    return X_out, y_out


def augment_merged_df(merged_df: pd.DataFrame, n: int) -> pd.DataFrame:

    permutation_list: list[int] = [i for i in range(1, n)]
    permutations: list[tuple[int]] = []

    for perm in itertools.permutations(permutation_list):
        add = [0] + list(perm)
        permutations.append(tuple(add))

    return add_permutations_merged_df(
        merged_df,
        n,
        permutations=permutations,
    )


def add_permutations_merged_df(
    merged_df: pd.DataFrame,
    n: int,
    permutations: list[tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """
    Add permutations to a merged dataframe.

    Parameters:
        merged_df: pd.DataFrame
            Dataframe to add permutations to

        n: int
            Number of entries per group

        permutations: list[tuple] | None
            List of tuples containing the indices to swap. Index j on i-th position
            means that the i-th row is swapped with the j-th row.
            If None, all permutations are added.

    Returns:
        pd.DataFrame
            Dataframe with added permutations
    """
    if permutations is None:
        permutations = list(itertools.permutations([i for i in range(n)]))

    out_dfs = []

    for perm in permutations:
        new_df = merged_df.copy()
        for i, j in enumerate(perm):
            new_df = swap_rows_merged_dataframe(new_df, i, j)

        out_dfs.append(new_df)

    return pd.concat(out_dfs, axis=0, ignore_index=True)


def drop_multi_cols(
    merged_df: pd.DataFrame, cols: list[str], n: int
) -> pd.DataFrame:
    """
    Drop multiindex columns from a dataframe, except the zeroth one.
    Parameters:
    merged_df: dataframe to drop duplicate columns from
    cols: columns to drop
    n: number of entries per group
    """
    merged_df = merged_df.copy()

    # Drop all except the zeroth one
    for col in cols:
        if col not in merged_df.columns:
            print(f"Warning: Column {col} not in dataframe")
            continue

        merged_df = merged_df.loc[
            :, merged_df.columns.drop([(col, i) for i in range(1, n)])
        ]

    return merged_df


def drop_padded_rows(
    df: pd.DataFrame,
    column: str,
    null_value: int = 0,
) -> pd.DataFrame:
    """
    Drop the rows where the zeroth column is `null_value`.

    Parameters:
        df:
            Dataframe to drop rows from
        column:
            Column to group by
        null_value:
            Value to drop rows by. Default is 0.

    Returns:
        dataframe with dropped rows
    """
    return df[df[column][0] != null_value].copy()
