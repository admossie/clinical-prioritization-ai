import joblib

preprocessor = joblib.load('models/preprocessor.joblib')

# Get all feature names expected by the preprocessor
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    # For older sklearn or custom preprocessors
    feature_names = preprocessor.transformers_[0][2] if hasattr(preprocessor, 'transformers_') else []

print('Expected feature names:')
print(feature_names)
