"""Dashboard 评估页。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


def render() -> None:
    st.title("Evaluation Panel")
    settings = st.session_state.get("_dashboard_settings")
    hybrid_search = st.session_state.get("_dashboard_hybrid_search")
    if settings is None or hybrid_search is None:
        st.info("Dashboard evaluation context not ready.")
        return

    backend = st.selectbox("Backend", ["custom", "ragas", "composite"], index=0)
    test_set_path = st.text_input("Golden Test Set", value="tests/fixtures/golden_test_set.json")

    if st.button("Run Evaluation"):
        try:
            if backend == "custom":
                evaluator = CustomEvaluator(settings)
            elif backend == "ragas":
                evaluator = RagasEvaluator(settings)
            else:
                evaluator = CompositeEvaluator([CustomEvaluator(settings), RagasEvaluator(settings)], settings=settings)

            report = EvalRunner(settings, hybrid_search, evaluator).run(test_set_path)
            st.success("evaluation finished")
            st.json(report["metrics"])
            st.json(report)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
