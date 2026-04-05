from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .schemas import TARGET_SOURCE_COLUMN, TARGET_COLUMN

def load_and_prepare_data(path:str)->pd.DataFrame:
    import numpy as np
    df = pd.read_csv(path).replace("?", np.nan)
    if TARGET_SOURCE_COLUMN in df.columns and TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN]=(df[TARGET_SOURCE_COLUMN]=="<30").astype(int)
        df=df.drop(columns=[TARGET_SOURCE_COLUMN])
    return df

def build_preprocessor(df: pd.DataFrame, target_col:str=TARGET_COLUMN):
    X=df.drop(columns=[target_col])
    numeric_cols=X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols=[c for c in X.columns if c not in numeric_cols]
    numeric_pipe=Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe=Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric_cols),("cat", categorical_pipe, categorical_cols)])

def align_to_preprocessor_input(df: pd.DataFrame, preprocessor)->pd.DataFrame:
    expected=[]
    for _,_,cols in preprocessor.transformers_:
        if isinstance(cols,list): expected.extend(cols)
        elif isinstance(cols,tuple): expected.extend(list(cols))
    aligned=df.copy()
    for col in expected:
        if col not in aligned.columns: aligned[col]=0
    return aligned[expected]
