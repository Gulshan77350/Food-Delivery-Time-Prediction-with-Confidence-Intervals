"""Smoke-test the Streamlit app headlessly (skipped until the pipeline has run)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "eta_bundle.joblib").exists(),
                                reason="run `python -m src.pipeline` first")


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)


def test_app_renders_test_order_mode():
    at = _app().run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Promised window" in labels and "Actual" in labels


def test_app_simulate_mode_produces_interval():
    at = _app().run()
    at.radio[0].set_value("Simulate a new order").run()
    assert not at.exception, at.exception
    window = next(m for m in at.metric if m.label == "Promised window").value
    lo, hi = (int(x) for x in window.replace(" min", "").split("–"))
    assert 0 < lo < hi < 120
