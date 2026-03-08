"""Dashboard 评估页占位实现。"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Evaluation Panel")
    st.info("H 阶段完成后，这里将支持运行 golden test set 评估。")
