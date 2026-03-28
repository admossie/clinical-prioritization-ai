from __future__ import annotations
import argparse, joblib, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier
from .preprocess import load_and_prepare_data, build_preprocessor
from .temporal_features import add_temporal_features
from .schemas import TARGET_COLUMN
from .cost_sensitive import optimize_threshold

def split_data(df: pd.DataFrame, target_col:str=TARGET_COLUMN):
    if "patient_nbr" in df.columns:
        X=df.drop(columns=[target_col]); y=df[target_col]; groups=df["patient_nbr"]
        splitter=GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx,test_idx=next(splitter.split(X,y,groups))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    train_df,test_df=train_test_split(df,test_size=0.25,stratify=df[target_col],random_state=42)
    return train_df.copy(),test_df.copy()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-path", required=True); args=p.parse_args()
    df=add_temporal_features(load_and_prepare_data(args.data_path))
    train_df,test_df=split_data(df); preprocessor=build_preprocessor(train_df)
    X_train,y_train=train_df.drop(columns=[TARGET_COLUMN]),train_df[TARGET_COLUMN]
    X_test,y_test=test_df.drop(columns=[TARGET_COLUMN]),test_df[TARGET_COLUMN]
    Xt_train=preprocessor.fit_transform(X_train); Xt_test=preprocessor.transform(X_test)
    models={"logreg":LogisticRegression(max_iter=1000,class_weight="balanced"),"xgb":XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.8,eval_metric="logloss",random_state=42)}
    results=[]; fitted={}
    for name, model in models.items():
        model.fit(Xt_train, y_train); probs=model.predict_proba(Xt_test)[:,1]
        results.append({"model":name,"pr_auc":float(average_precision_score(y_test, probs))}); fitted[name]=(model, probs)
    results_df=pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name=results_df.iloc[0]["model"]; best_model,best_probs=fitted[best_name]
    best_threshold, threshold_table=optimize_threshold(y_test, best_probs)
    Path("models").mkdir(exist_ok=True); Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib"); joblib.dump(preprocessor, "models/preprocessor.joblib")
    results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
    pd.DataFrame([best_threshold]).to_csv("outputs/tables/best_threshold.csv", index=False)
    threshold_table.to_csv("outputs/tables/threshold_sweep.csv", index=False)
    scored=test_df.copy(); scored["risk_score"]=best_probs; scored.to_csv("outputs/tables/test_scored.csv", index=False)
    # print(results_df)
if __name__=="__main__": main()
