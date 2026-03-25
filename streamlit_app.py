"""
Continual Incident Intelligence — dashboard.

Run from repo root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from incident_cl.config import ExperimentConfig
from incident_cl.experiment import load_results, run_comparison, save_results

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "results.json"


def matrix_to_df(acc_matrix: list[dict[int, float]], name: str) -> pd.DataFrame:
    rows = []
    for train_era, evals in enumerate(acc_matrix):
        for test_era in sorted(evals.keys()):
            rows.append(
                {
                    "mode": name,
                    "train_through_era": train_era,
                    "test_era": test_era,
                    "accuracy": evals[test_era],
                }
            )
    return pd.DataFrame(rows)


def heatmap_fig(df: pd.DataFrame, title: str) -> go.Figure:
    pivot = df.pivot(
        index="train_through_era",
        columns="test_era",
        values="accuracy",
    )
    fig = px.imshow(
        pivot.values,
        x=[f"E{c}" for c in pivot.columns],
        y=[f"After E{r}" for r in pivot.index],
        zmin=0,
        zmax=1,
        color_continuous_scale="Viridis",
        title=title,
        aspect="auto",
    )
    fig.update_layout(height=400)
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Continual Incident CL",
        layout="wide",
    )
    st.title("Continual incident intelligence (synthetic era benchmark)")
    st.caption(
        "Streaming ops-style incidents: dominant failure families shift by era; "
        "legacy types stay rare. Compare naive fine-tuning vs bounded replay."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run experiment (CPU/GPU)", type="primary"):
            with st.spinner("Training naive + replay…"):
                cfg = ExperimentConfig()
                payload = run_comparison(cfg)
                ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
                save_results(payload, ARTIFACT)
            st.success("Done.")
    with col_b:
        uploaded = st.file_uploader("Or load results JSON", type=["json"])

    data = None
    if uploaded is not None:
        data = json.loads(uploaded.read().decode("utf-8"))
    elif ARTIFACT.is_file():
        data = load_results(ARTIFACT)

    if data is None:
        st.info(
            f"Click **Run experiment** or upload a JSON file. "
            f"CLI: `python -m incident_cl.cli` writes `{ARTIFACT}`."
        )
        return

    naive = data["naive"]
    replay = data["replay"]

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Naive (new stream only)")
        st.json(naive["summary"])
    with s2:
        st.subheader("Replay (bounded buffer)")
        st.json(replay["summary"])

    df_n = matrix_to_df(naive["acc_matrix"], "naive")
    df_r = matrix_to_df(replay["acc_matrix"], "replay")
    df_all = pd.concat([df_n, df_r], ignore_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            heatmap_fig(df_n, "Naive — accuracy on era test after training through era"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            heatmap_fig(df_r, "Replay — accuracy on era test after training through era"),
            use_container_width=True,
        )

    st.subheader("Backward stability: accuracy on era 0 test over time")
    e0 = df_all[df_all["test_era"] == 0]
    fig2 = px.line(
        e0,
        x="train_through_era",
        y="accuracy",
        color="mode",
        markers=True,
        title="Higher at the end = less forgetting of legacy incidents",
    )
    fig2.update_yaxes(range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Replay buffer composition (final training)")
    if replay.get("buffer_history"):
        last = replay["buffer_history"][-1]
        st.bar_chart(
            pd.Series(last, name="samples").sort_index(),
        )


if __name__ == "__main__":
    main()
