from __future__ import annotations
import pandas as pd

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    if "patient_nbr" not in df.columns:
        return df
    df=df.sort_values([c for c in ["patient_nbr","encounter_id"] if c in df.columns])
    grp=df.groupby("patient_nbr", group_keys=False)
    df["encounter_number"]=grp.cumcount()+1
    df["prior_encounters"]=df["encounter_number"]-1
    for col in [c for c in ["number_inpatient","number_outpatient","number_emergency"] if c in df.columns]:
        df[f"prior_{col}_sum"]=grp[col].cumsum()-df[col]
        df[f"prior_{col}_mean"]=grp[col].apply(lambda s: s.shift(1).expanding().mean())
    if set(["number_inpatient","number_outpatient","number_emergency"]).issubset(df.columns):
        df["prior_total_visits"]=(df["prior_number_inpatient_sum"].fillna(0)+df["prior_number_outpatient_sum"].fillna(0)+df["prior_number_emergency_sum"].fillna(0))
    if "number_diagnoses" in df.columns: df["diag_delta"]=grp["number_diagnoses"].diff().fillna(0)
    if "num_medications" in df.columns: df["med_delta"]=grp["num_medications"].diff().fillna(0)
    if "target" in df.columns:
        df["prior_positive_count"]=(grp["target"].cumsum()-df["target"]).fillna(0)
        df["ever_prior_positive"]=(df["prior_positive_count"]>0).astype(int)
    else:
        df["prior_positive_count"]=0
        df["ever_prior_positive"]=0
    for c in [x for x in df.columns if x.startswith("prior_")]+["diag_delta","med_delta"]:
        if c in df.columns: df[c]=df[c].fillna(0)
    return df
