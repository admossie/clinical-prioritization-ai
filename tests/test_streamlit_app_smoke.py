from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_and_predicts() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run(timeout=120)

    assert not at.exception
    assert any(button.label == "Predict risk" for button in at.button)

    predict_button = next(
        button for button in at.button if button.label == "Predict risk"
    )
    predict_button.click()
    at.run(timeout=120)

    metric_labels = {metric.label for metric in at.metric}
    assert "Estimated risk" in metric_labels
    assert "Priority tier" in metric_labels
    assert any(
        "How to read this result" in getattr(markdown, "value", "")
        for markdown in at.markdown
    )
    assert any(
        "Recommended action:" in getattr(info_box, "value", "") for info_box in at.info
    )
