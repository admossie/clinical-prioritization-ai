from pathlib import Path

import joblib
import numpy as np

from src.preprocess import build_preprocessor, load_and_prepare_data, transform_with_feature_names
from src.schemas import TARGET_COLUMN
from src.temporal_features import add_temporal_features

DATA_PATH = Path("data/raw/diabetic_data.csv")
MODEL_PATH = Path("models/best_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


def test_prepare_and_fit_preprocessor_smoke():
    df = add_temporal_features(load_and_prepare_data(str(DATA_PATH))).head(200).copy()

    assert not df.empty
    assert TARGET_COLUMN in df.columns

    preprocessor = build_preprocessor(df)
    x = df.drop(columns=[TARGET_COLUMN])
    xt = preprocessor.fit_transform(x)

    assert xt.shape[0] == len(x)


def test_saved_model_artifacts_score_probabilities():
    df = add_temporal_features(load_and_prepare_data(str(DATA_PATH))).head(25).copy()
    x = df.drop(columns=[TARGET_COLUMN])

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    xt = transform_with_feature_names(x, preprocessor)
    probs = model.predict_proba(xt)[:, 1]

    assert len(probs) == len(x)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
