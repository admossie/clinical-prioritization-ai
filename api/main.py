from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.inference import (
    DEFAULT_DEMO_INPUTS,
    load_model_metadata,
    load_pipeline,
    score_batch_payloads,
    score_patient_payload,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    age: str = Field(default=DEFAULT_DEMO_INPUTS["age"])
    gender: str = Field(default=DEFAULT_DEMO_INPUTS["gender"])
    race: str = Field(default=DEFAULT_DEMO_INPUTS["race"])
    time_in_hospital: int = Field(
        default=DEFAULT_DEMO_INPUTS["time_in_hospital"], ge=1, le=14
    )
    num_lab_procedures: int = Field(
        default=DEFAULT_DEMO_INPUTS["num_lab_procedures"], ge=1, le=100
    )
    num_procedures: int = Field(
        default=DEFAULT_DEMO_INPUTS["num_procedures"], ge=0, le=6
    )
    num_medications: int = Field(
        default=DEFAULT_DEMO_INPUTS["num_medications"], ge=1, le=40
    )
    number_diagnoses: int = Field(
        default=DEFAULT_DEMO_INPUTS["number_diagnoses"], ge=1, le=16
    )
    number_outpatient: int = Field(
        default=DEFAULT_DEMO_INPUTS["number_outpatient"], ge=0, le=20
    )
    number_emergency: int = Field(
        default=DEFAULT_DEMO_INPUTS["number_emergency"], ge=0, le=20
    )
    number_inpatient: int = Field(
        default=DEFAULT_DEMO_INPUTS["number_inpatient"], ge=0, le=20
    )
    admission_type_id: int = Field(
        default=DEFAULT_DEMO_INPUTS["admission_type_id"], ge=1, le=8
    )
    discharge_disposition_id: int = Field(
        default=DEFAULT_DEMO_INPUTS["discharge_disposition_id"], ge=1, le=30
    )
    admission_source_id: int = Field(
        default=DEFAULT_DEMO_INPUTS["admission_source_id"], ge=1, le=25
    )
    prior_inpatient: int = Field(
        default=DEFAULT_DEMO_INPUTS["prior_inpatient"], ge=0, le=20
    )
    prior_outpatient: int = Field(
        default=DEFAULT_DEMO_INPUTS["prior_outpatient"], ge=0, le=20
    )
    prior_emergency: int = Field(
        default=DEFAULT_DEMO_INPUTS["prior_emergency"], ge=0, le=20
    )
    prior_positive_count: int = Field(
        default=DEFAULT_DEMO_INPUTS["prior_positive_count"], ge=0, le=10
    )
    diag_delta: int = Field(default=DEFAULT_DEMO_INPUTS["diag_delta"], ge=-10, le=10)
    med_delta: int = Field(default=DEFAULT_DEMO_INPUTS["med_delta"], ge=-20, le=20)
    encounter_id: int = 999999
    patient_nbr: int = 9999


class BatchPredictRequest(BaseModel):
    patients: list[PredictRequest] = Field(min_length=1)


app = FastAPI(
    title="AI Care Prioritization API",
    description=(
        "Minimal production-style inference API for the readmission "
        "prioritization demo."
    ),
    version="1.0.6",
)


def _request_to_dict(request: PredictRequest) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "AI Care Prioritization API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch_predict": "/batch_predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _, _, pipeline_source = load_pipeline()
    metadata = load_model_metadata(
        use_saved_artifacts=pipeline_source == "saved-artifacts"
    )
    return {
        "status": "ok",
        "service": "ai-care-prioritization-api",
        "model_version": metadata.get("version", "unversioned"),
        "artifact_source": metadata.get("artifact_source", pipeline_source),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    return score_patient_payload(_request_to_dict(request))


@app.post("/batch_predict")
def batch_predict(request: BatchPredictRequest) -> dict[str, Any]:
    patients = [_request_to_dict(patient) for patient in request.patients]
    return score_batch_payloads(patients)


if __name__ == "__main__":
    host = os.getenv("CARE_API_HOST", "127.0.0.1")
    port = int(os.getenv("CARE_API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
