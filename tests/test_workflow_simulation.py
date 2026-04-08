import pandas as pd

from src.workflow_simulation import (
    hospital_roi,
    run_workflow_scenarios,
    simulate_workflow,
)


def test_workflow_helpers_return_expected_shapes():
    df = pd.DataFrame(
        {
            "risk_score": [0.95, 0.80, 0.30, 0.10],
            "target": [1, 0, 1, 0],
        }
    )

    summary = simulate_workflow(df, capacity=2)
    roi = hospital_roi(df, capacity=2, intervention_effectiveness=0.50)
    scenarios = run_workflow_scenarios(df, capacities=(1, 2))

    assert summary["patients_reviewed"] == 2
    assert 0.0 <= summary["capture_rate_in_queue"] <= 1.0
    assert roi["patients_targeted"] == 2
    assert roi["net_roi"] == roi["savings"] - roi["cost"]
    assert list(scenarios["capacity"]) == [1, 2]
