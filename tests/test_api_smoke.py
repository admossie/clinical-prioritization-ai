from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_api_health_and_predict() -> None:
    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["status"] == "ok"
    assert "model_version" in health_payload

    predict_response = client.post(
        "/predict",
        json={
            "age": "[70-80)",
            "gender": "Female",
            "race": "Caucasian",
            "time_in_hospital": 7,
            "num_medications": 20,
            "number_diagnoses": 10,
            "prior_inpatient": 2,
            "prior_outpatient": 1,
            "prior_emergency": 1,
            "prior_positive_count": 1,
        },
    )
    assert predict_response.status_code == 200

    payload = predict_response.json()
    assert 0.0 <= payload["risk"] <= 1.0
    assert payload["tier"] in {"Low", "Medium", "High"}
    assert 0.0 <= payload["risk_percentile"] <= 100.0
    assert payload["artifact_source"] in {"saved-artifacts", "fallback-demo"}


def test_api_batch_predict() -> None:
    batch_response = client.post(
        "/batch_predict",
        json={
            "patients": [
                {
                    "age": "[70-80)",
                    "gender": "Female",
                    "race": "Caucasian",
                    "time_in_hospital": 7,
                    "num_medications": 20,
                    "number_diagnoses": 10,
                    "prior_inpatient": 2,
                },
                {
                    "age": "[40-50)",
                    "gender": "Male",
                    "race": "Other",
                    "time_in_hospital": 2,
                    "num_medications": 6,
                    "number_diagnoses": 4,
                    "prior_inpatient": 0,
                },
            ]
        },
    )
    assert batch_response.status_code == 200

    payload = batch_response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["tier"] in {"Low", "Medium", "High"}
