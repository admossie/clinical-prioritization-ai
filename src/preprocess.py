from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .schemas import TARGET_SOURCE_COLUMN, TARGET_COLUMN


def load_and_prepare_data(path: str) -> pd.DataFrame:
    import numpy as np

    df = pd.read_csv(path).replace("?", np.nan)
    if TARGET_SOURCE_COLUMN in df.columns and TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_SOURCE_COLUMN] == "<30").astype(int)
        df = df.drop(columns=[TARGET_SOURCE_COLUMN])
    return df


def build_preprocessor(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_all_missing = [c for c in numeric_cols if X[c].isna().all()]
    categorical_all_missing = [c for c in categorical_cols if X[c].isna().all()]
    numeric_observed = [c for c in numeric_cols if c not in numeric_all_missing]
    categorical_observed = [
        c for c in categorical_cols if c not in categorical_all_missing
    ]

    transformers = []
    if numeric_observed:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_observed,
            )
        )
    if numeric_all_missing:
        transformers.append(
            (
                "num_missing",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=0,
                                keep_empty_features=True,
                            ),
                        )
                    ]
                ),
                numeric_all_missing,
            )
        )
    if categorical_observed:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_observed,
            )
        )
    if categorical_all_missing:
        transformers.append(
            (
                "cat_missing",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="missing",
                                keep_empty_features=True,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_all_missing,
            )
        )
    return ColumnTransformer(transformers)


def align_to_preprocessor_input(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    expected = []
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            expected.extend(cols)
        elif isinstance(cols, tuple):
            expected.extend(list(cols))
    aligned = df.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[expected]


def transform_with_feature_names(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    Xt = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(Xt, "tocoo"):
        return pd.DataFrame.sparse.from_spmatrix(
            Xt, index=df.index, columns=feature_names
        )
    return pd.DataFrame(Xt, index=df.index, columns=feature_names)
