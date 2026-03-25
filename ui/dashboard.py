"""
Incident Memory Engine — professional Streamlit control room (HTTP-only client).

Run: ``uvicorn incident_memory_engine.api.app:app --port 8000`` then
``streamlit run ui/dashboard.py`` from the repo root so paths resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
DEFAULT_GITHUB_JSON = _ROOT / "artifacts" / "github_issues.json"
FORGETTING_CURVE_PNG = _ROOT / "artifacts" / "forgetting_curve.png"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_PERSIST = "artifacts/last_run.json"

# Plotly presentation defaults
_CHART_FONT = dict(family="Segoe UI, Inter, system-ui, sans-serif", size=13, color="#0f172a")
_LAYOUT_BASE = dict(
    template="plotly_white",
    font=_CHART_FONT,
    paper_bgcolor="#f8fafc",
    plot_bgcolor="#ffffff",
    margin=dict(l=64, r=48, t=80, b=56),
    colorway=["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#db2777"],
    hoverlabel=dict(font_size=13, font_family="Segoe UI"),
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
        h1 { font-weight: 700 !important; letter-spacing: -0.02em; color: #0f172a !important; }
        .stApp { background: linear-gradient(180deg, #f1f5f9 0%, #f8fafc 32%, #ffffff 100%); }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }
        div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.8rem !important; }
        .block-container { padding-top: 1.5rem; max-width: 1400px; }
        [data-testid="stTabs"] button { font-weight: 600 !important; letter-spacing: -0.01em; }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: #1d4ed8 !important;
            border-bottom-color: #2563eb !important;
        }
        .stTextArea textarea, .stTextInput input {
            border-radius: 10px !important;
            border-color: #cbd5e1 !important;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _headers() -> dict[str, str]:
    key = st.session_state.get("api_key") or ""
    if key:
        return {"X-API-Key": key}
    return {}


def _client(base: str, timeout: float = 180.0) -> httpx.Client:
    return httpx.Client(
        base_url=base.rstrip("/"),
        timeout=timeout,
        headers=_headers(),
    )


def _fetch_health(base: str) -> None:
    try:
        with _client(base, 30.0) as c:
            r = c.get("/health")
            r.raise_for_status()
            st.session_state["health"] = r.json()
    except Exception as e:
        st.session_state["health"] = {"error": str(e)}


def _refresh_metrics_alert(base: str) -> None:
    with _client(base) as c:
        rm = c.get("/metrics")
        rm.raise_for_status()
        ra = c.get("/forgetting-alert")
        ra.raise_for_status()
        st.session_state["metrics"] = rm.json()
        st.session_state["alert"] = ra.json()


def _envelope_data(resp: httpx.Response) -> dict:
    j = resp.json()
    if isinstance(j, dict) and j.get("ok") is True and "data" in j:
        return j["data"]
    return j


def _insight_inner(raw: dict) -> dict:
    if isinstance(raw, dict) and raw.get("ok") is True and isinstance(raw.get("data"), dict):
        return raw["data"]
    return raw


def _render_predict_summary(data: dict) -> None:
    if not isinstance(data, dict):
        return
    if "predicted_class" in data and "confidence" in data:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted class", int(data["predicted_class"]))
        with c2:
            st.metric("Confidence", f"{float(data['confidence']):.4f}")
    with st.expander("Raw JSON", expanded=False):
        st.json(data)


def _render_similar_table(data: dict) -> None:
    if not isinstance(data, dict):
        return
    matches = data.get("matches") or []
    if not matches:
        st.warning("No matches — train or ingest data first so the buffer / index has incidents.")
        with st.expander("Raw JSON", expanded=False):
            st.json(data)
        return
    rows = []
    for i, m in enumerate(matches, start=1):
        rows.append(
            {
                "#": i,
                "label": m.get("label"),
                "era": m.get("era"),
                "distance": round(float(m.get("distance", 0.0)), 5),
                "similarity": round(float(m.get("similarity_score", 0.0)), 4),
                "rank": round(float(m.get("rank_score", 0.0)), 4) if m.get("rank_score") is not None else None,
                "tier": m.get("memory_tier", ""),
                "type": (m.get("incident_type") or "")[:56],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Raw JSON", expanded=False):
        st.json(data)


def _render_insight_panel(raw: dict) -> None:
    d = _insight_inner(raw)
    pred = d.get("prediction") or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        cid = pred.get("class_id")
        st.metric("Predicted class", str(int(cid)) if cid is not None else "—")
    with c2:
        cn = str(pred.get("class_name") or "—")
        st.metric("Label name", cn[:22] + ("…" if len(cn) > 22 else ""))
    with c3:
        conf = pred.get("confidence")
        st.metric("Confidence", f"{float(conf):.4f}" if conf is not None else "—")
    fix = (d.get("suggested_fix") or "").strip()
    if fix:
        st.success(fix[:8000])
    nv = d.get("neighbor_vote") or {}
    if nv:
        st.caption("Neighbor label vote")
        st.json(nv)
    neigh = d.get("similar_incidents") or []
    if neigh:
        nr = []
        for n in neigh:
            nr.append(
                {
                    "label": n.get("label"),
                    "era": n.get("era"),
                    "distance": round(float(n.get("distance", 0.0)), 5),
                    "similarity": round(float(n.get("similarity_score", 0.0)), 4),
                    "weight": round(float(n.get("weight", 0.0)), 4),
                    "tier": n.get("memory_tier", ""),
                }
            )
        st.caption("Attributed neighbors")
        st.dataframe(pd.DataFrame(nr), use_container_width=True, hide_index=True)
    exp = d.get("explainability") or {}
    if exp:
        with st.expander("Explainability (neighbor weights)", expanded=False):
            st.json(exp)
    fw = d.get("forgetting_warning")
    if fw:
        with st.expander("Forgetting context", expanded=False):
            st.json(fw)
    llm_part = d.get("llm")
    if llm_part:
        with st.expander("LLM block", expanded=False):
            st.json(llm_part)
    with st.expander("Full response JSON", expanded=False):
        st.json(raw)


def _render_cl_status(metrics: dict) -> None:
    """Surface replay + EWC from ``/metrics`` (newer APIs) or hint when absent."""
    cl = metrics.get("continual_learning")
    if isinstance(cl, dict):
        rc = cl.get("replay_capacity")
        rbr = cl.get("replay_batch_ratio")
        ew = float(cl.get("ewc_lambda") or 0)
        rdy = bool(cl.get("ewc_consolidation_ready"))
        rbr_s = f"{float(rbr):.2f}" if rbr is not None else "—"
        if ew > 0:
            ewc_s = f"EWC **on** (λ={ew}) · anchor **{'ready' if rdy else 'after first era close'}**"
        else:
            ewc_s = (
                "EWC **off** — set API env **`IME_EWC_LAMBDA`** (e.g. `25`) or "
                "`EngineConfig.ewc_lambda` for weight consolidation alongside replay"
            )
        st.info(
            f"**Continual learning:** replay cap **{rc}**, batch mix **{rbr_s}** · {ewc_s}"
        )
    else:
        st.caption(
            "Tip: upgrade the API to expose `continual_learning` on `/metrics` for replay/EWC details."
        )


def _kpi_strip(metrics: dict, alert: dict | None) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("BWT", f"{metrics.get('bwt', 0):.4f}")
    with c2:
        st.metric("Mean forgetting", f"{metrics.get('mean_forgetting', 0):.4f}")
    with c3:
        cla = metrics.get("current_legacy_accuracy")
        st.metric(
            "Legacy (E0) now",
            f"{cla:.4f}" if cla is not None else "—",
        )
    with c4:
        peak = metrics.get("peak_legacy_accuracy")
        st.metric("Legacy peak", f"{peak:.4f}" if peak is not None else "—")
    with c5:
        risk = (alert or {}).get("risk_level", "—")
        st.metric("Alert risk", str(risk).upper())


def _fig_accuracy_matrix(metrics: dict) -> go.Figure | None:
    mat = metrics.get("accuracy_matrix") or []
    if not mat:
        return None
    rows = []
    for train_era, evals in enumerate(mat):
        for test_era_s, acc in evals.items():
            rows.append(
                {
                    "After training through era": train_era,
                    "Evaluation era": int(test_era_s),
                    "Accuracy": float(acc),
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(
        index="After training through era",
        columns="Evaluation era",
        values="Accuracy",
    )
    z = pivot.values
    text = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"E{c}" for c in pivot.columns],
            y=[f"Row {r}" for r in pivot.index],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Acc", tickformat=".0%"),
        )
    )
    fig.update_layout(
        title=dict(
            text="<b>Accuracy matrix</b><br><sup>Rows: after closing each training era · Columns: holdout test era</sup>",
            font=dict(size=16, family="DM Sans, sans-serif", color="#0f172a"),
        ),
        xaxis_title="Test era",
        yaxis_title="Train-through (matrix row)",
        **_LAYOUT_BASE,
        height=440,
    )
    return fig


def _fig_forgetting_curves(metrics: dict) -> go.Figure | None:
    mat = metrics.get("accuracy_matrix") or []
    if not mat:
        return None
    test_eras = sorted({int(k) for row in mat for k in row.keys()}, key=int)
    fig = go.Figure()
    for te in test_eras:
        xs: list[int] = []
        ys: list[float] = []
        for ti, row in enumerate(mat):
            sk = str(te)
            if sk in row:
                xs.append(ti)
                ys.append(float(row[sk]))
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"Holdout E{te}",
                    line=dict(width=2.5),
                    marker=dict(size=9),
                )
            )
    fig.update_layout(
        title=dict(
            text="<b>Forgetting curves</b><br><sup>Per test-era accuracy as training progresses (rows of the matrix)</sup>",
            font=dict(size=16, family="DM Sans, sans-serif"),
        ),
        xaxis_title="Matrix row index (after each era close)",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT_BASE,
        height=420,
    )
    return fig


def _fig_legacy_history(metrics: dict) -> go.Figure | None:
    hist = metrics.get("legacy_accuracy_history") or []
    if not hist:
        return None
    fig = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(hist))),
                y=hist,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=10, color="#1d4ed8"),
                name="Era-0 holdout",
            )
        ]
    )
    peak = metrics.get("peak_legacy_accuracy")
    if peak is not None:
        fig.add_hline(
            y=float(peak),
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Peak {peak:.2f}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=dict(text="<b>Legacy accuracy trajectory</b>", font=dict(size=16)),
        xaxis_title="Era close index",
        yaxis_title="Accuracy on era-0 test set",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        showlegend=False,
        **_LAYOUT_BASE,
        height=360,
    )
    return fig


def _fig_buffer_composition(bh: dict) -> go.Figure | None:
    if not bh:
        return None
    s = pd.Series(bh, name="samples").sort_index()
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f"E{k}" for k in s.index],
                y=s.values,
                marker_color="#059669",
                text=s.values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Replay buffer · samples per era</b>", font=dict(size=16)),
        xaxis_title="Era",
        yaxis_title="Count",
        **_LAYOUT_BASE,
        height=340,
    )
    return fig


def _fig_drift_proxy(metrics: dict) -> go.Figure | None:
    mat = metrics.get("accuracy_matrix") or []
    if not mat:
        return None
    rows = []
    for train_era, evals in enumerate(mat):
        for test_era_s, acc in evals.items():
            rows.append(
                {
                    "train_through": train_era,
                    "test_era": int(test_era_s),
                    "accuracy": float(acc),
                }
            )
    df = pd.DataFrame(rows)
    drift_df = (
        df.groupby("test_era", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mean_accuracy"})
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=drift_df["test_era"],
                y=drift_df["mean_accuracy"],
                mode="lines+markers",
                line=dict(color="#d97706", width=3),
                marker=dict(size=11, color="#b45309"),
                fill="tozeroy",
                fillcolor="rgba(217, 119, 6, 0.15)",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="<b>Drift proxy</b><br><sup>Mean matrix accuracy per test era (summary view)</sup>",
            font=dict(size=16),
        ),
        xaxis_title="Test era",
        yaxis_title="Mean accuracy",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        **_LAYOUT_BASE,
        height=360,
    )
    return fig


def _render_alert_card(alert: dict | None) -> None:
    if not alert:
        return
    level = str(alert.get("risk_level", "low")).lower()
    msg = alert.get("message") or alert.get("explanation", "")
    drop = alert.get("drop_percentage")
    conf = alert.get("confidence")
    parts = [msg]
    if drop is not None:
        parts.append(f"Drop from peak: **{drop}%**")
    if conf is not None:
        parts.append(f"Model confidence in alert: **{conf}**")
    if alert.get("affected_eras"):
        parts.append(f"Affected eras: `{alert['affected_eras']}`")
    text = " · ".join(parts)
    if level == "high":
        st.error(f"**Forgetting risk — HIGH**  \n{text}")
    elif level == "medium":
        st.warning(f"**Forgetting risk — MEDIUM**  \n{text}")
    else:
        st.success(f"**Forgetting risk — LOW**  \n{text}")


def _render_analytics_dashboard(
    metrics: dict,
    alert: dict | None,
    *,
    chart_scope: str = "main",
) -> None:
    """``chart_scope`` must differ per Streamlit render path (e.g. Home vs Analytics) so plotly_chart IDs stay unique."""
    _render_cl_status(metrics)
    _kpi_strip(metrics, alert)
    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_m = _fig_accuracy_matrix(metrics)
        if fig_m:
            st.plotly_chart(
                fig_m, use_container_width=True, key=f"{chart_scope}_acc_matrix"
            )
        fig_f = _fig_forgetting_curves(metrics)
        if fig_f:
            st.plotly_chart(
                fig_f, use_container_width=True, key=f"{chart_scope}_forgetting"
            )
    with c2:
        fig_l = _fig_legacy_history(metrics)
        if fig_l:
            st.plotly_chart(
                fig_l, use_container_width=True, key=f"{chart_scope}_legacy"
            )
        fig_d = _fig_drift_proxy(metrics)
        if fig_d:
            st.plotly_chart(fig_d, use_container_width=True, key=f"{chart_scope}_drift")
    fig_b = _fig_buffer_composition(metrics.get("buffer_era_histogram") or {})
    if fig_b:
        st.plotly_chart(fig_b, use_container_width=True, key=f"{chart_scope}_buffer")


def main() -> None:
    st.set_page_config(
        page_title="Incident Memory Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    st.title("Incident Memory Engine")
    st.caption(
        "Continual learning control room · FastAPI backend · all ML via HTTP · "
        "Data: public GitHub issues (kubernetes, prometheus, grafana) via GitHub REST Search API"
    )

    base = st.sidebar.text_input("API base URL", value=DEFAULT_API)
    st.session_state["api_key"] = st.sidebar.text_input(
        "X-API-Key (if server uses IME_API_KEYS)",
        value=st.session_state.get("api_key", ""),
        type="password",
    )
    st.sidebar.divider()
    if st.sidebar.button("Health check", use_container_width=True):
        _fetch_health(base)
    if st.sidebar.button("Reset engine", use_container_width=True):
        try:
            with _client(base) as c:
                r = c.post("/engine/reset")
                r.raise_for_status()
            st.sidebar.success("Engine reset.")
            _fetch_health(base)
            _refresh_metrics_alert(base)
        except Exception as e:
            st.sidebar.error(str(e))
    if st.sidebar.button("Refresh metrics & alert", use_container_width=True):
        try:
            _refresh_metrics_alert(base)
            st.sidebar.success("Updated.")
        except Exception as e:
            st.sidebar.error(str(e))

    if "health" in st.session_state:
        h = st.session_state["health"]
        if h.get("error"):
            st.sidebar.error(h["error"])
        else:
            cap = (
                f"Buffer **{h.get('buffer_size', '—')}** · "
                f"encoder **{h.get('encoder_kind', '—')}** · "
                f"dim **{h.get('model_in_dim', '—')}**"
            )
            wl = h.get("ewc_lambda")
            if wl is not None and float(wl) > 0:
                active = bool(h.get("ewc_active"))
                cap += f" · EWC λ={float(wl)} ({'anchored' if active else 'pending'})"
            st.sidebar.caption(cap)
        with st.sidebar.expander("Raw health JSON"):
            st.json(h)

    tab_home, tab_data, tab_train, tab_analytics, tab_explore = st.tabs(
        [
            "Home",
            "Data pipeline",
            "Training lab",
            "Analytics",
            "Predict & similar",
        ]
    )

    with tab_home:
        st.markdown(
            "### Welcome\n"
            "Use **Data pipeline** to download GitHub issues and replay the full CL experiment "
            "on the server. **Training lab** supports synthetic or file-backed incremental steps. "
            "**Analytics** shows publication-style charts from live `/metrics`, plus replay / optional **EWC** status. "
            "Enable EWC on the API with env **`IME_EWC_LAMBDA`** (see README)."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            steps = st.number_input("Simulation steps / era", 10, 500, 80, 10)
            persist_sim = st.checkbox("Write metrics JSON to disk", key="sim_persist")
            sim_path_in = st.text_input(
                "Persist path", value=DEFAULT_PERSIST, key="sim_path", disabled=not persist_sim
            )
            if st.button("Run full synthetic simulation", type="primary"):
                with st.spinner("Running multi-era simulation…"):
                    try:
                        body: dict = {"steps_per_era": int(steps), "num_eras": None}
                        if persist_sim and sim_path_in:
                            body["persist_path"] = sim_path_in
                        with _client(base, 600.0) as c:
                            r = c.post("/simulation/run", json=body)
                            r.raise_for_status()
                            payload = r.json()
                            st.session_state["last_sim"] = payload
                            st.session_state["metrics"] = payload.get("metrics", {})
                            ra = c.get("/forgetting-alert")
                            ra.raise_for_status()
                            st.session_state["alert"] = ra.json()
                        st.success("Simulation complete.")
                    except Exception as e:
                        st.error(str(e))
        with col_b:
            st.info(
                "**Pro workflow:** Data pipeline → Download JSON → Replay experiment → "
                "open Analytics (or refresh metrics)."
            )
        metrics = st.session_state.get("metrics")
        if metrics is None and st.session_state.get("last_sim"):
            metrics = st.session_state["last_sim"].get("metrics")
        alert = st.session_state.get("alert")
        _render_alert_card(alert)
        if metrics:
            _render_analytics_dashboard(metrics, alert, chart_scope="home")
        else:
            st.info("Connect to the API and run a simulation or experiment to see charts.")

        if FORGETTING_CURVE_PNG.is_file():
            with st.expander("Static PNG from last `scripts/run_real_experiment.py` run"):
                st.image(str(FORGETTING_CURVE_PNG), use_container_width=True)

    with tab_data:
        st.subheader("1 · Download public GitHub issues")
        st.caption(
            "Sources: **kubernetes/kubernetes**, **prometheus/prometheus**, **grafana/grafana** "
            "(open-source repos on github.com). Uses GitHub Search API `is:issue`."
        )
        per_dl = st.number_input("Issues per repo / era", 50, 500, 300, 10, key="dl_per")
        out_rel = st.text_input(
            "Server-relative output path",
            value="artifacts/github_issues.json",
            help="Relative to the API server process working directory",
        )
        if st.button("POST /data/github/download", type="primary"):
            with st.spinner("Fetching from GitHub (can take ~1–2 min)…"):
                try:
                    with _client(base, 600.0) as c:
                        r = c.post(
                            "/data/github/download",
                            json={"per_era": int(per_dl), "output_path": out_rel},
                        )
                        r.raise_for_status()
                        data = _envelope_data(r)
                    st.success(
                        f"Saved **{data.get('sample_count', '?')}** samples → `{data.get('output_path')}`"
                    )
                    st.json(data)
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("2 · Replay full experiment on the live engine")
        st.caption(
            "Resets (optional), trains era-by-era, closes each era. Updates `/metrics` immediately."
        )
        data_path_srv = st.text_input(
            "Server path to JSON",
            value="artifacts/github_issues.json",
            key="replay_path",
        )
        reset_first = st.checkbox("Reset engine before replay", value=True, key="replay_reset")
        chunk_r = st.slider("Train chunk size", 8, 128, 64, key="replay_chunk")
        if st.button("POST /experiment/github-replay", type="primary"):
            with st.spinner("Training all eras (several minutes for ~900 samples)…"):
                try:
                    with _client(base, 900.0) as c:
                        r = c.post(
                            "/experiment/github-replay",
                            json={
                                "data_path": data_path_srv,
                                "reset_engine_first": reset_first,
                                "chunk_size": int(chunk_r),
                            },
                        )
                        r.raise_for_status()
                        result = _envelope_data(r)
                    st.session_state["metrics"] = result.get("metrics", {})
                    st.session_state["alert"] = result.get("forgetting_alert", {})
                    st.success(
                        f"Trained **{result.get('samples_trained', '?')}** samples · "
                        f"eras **{result.get('eras_closed', [])}**"
                    )
                    with st.expander("Full API response"):
                        st.json(result)
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("3 · Status")
        if st.button("GET /data/status"):
            try:
                with _client(base) as c:
                    r = c.get("/data/status")
                    r.raise_for_status()
                    st.json(r.json())
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.subheader("Optional · Train one era from live GitHub (no JSON file)")
        repos_in = st.text_input("Repos (comma-separated owner/name)", "kubernetes/kubernetes")
        era_gh = st.number_input("Era id", 0, 10, 0, key="ingh_era")
        per_repo = st.number_input("Per repo", 10, 500, 100, key="ingh_per")
        if st.button("POST /ingest/github"):
            with st.spinner("Fetching & training…"):
                try:
                    repos = [x.strip() for x in repos_in.split(",") if x.strip()]
                    with _client(base, 600.0) as c:
                        r = c.post(
                            "/ingest/github",
                            json={
                                "repos": repos,
                                "era": int(era_gh),
                                "per_repo": int(per_repo),
                            },
                        )
                        r.raise_for_status()
                        st.json(_envelope_data(r))
                    _refresh_metrics_alert(base)
                except Exception as e:
                    st.error(str(e))

    with tab_train:
        src = st.radio(
            "Batch source",
            ("Synthetic API batch", "Local github_issues.json (via API)"),
            horizontal=True,
        )
        st.session_state["github_json_path"] = st.text_input(
            "Path to github_issues.json (this machine — for local batching only)",
            value=str(st.session_state.get("github_json_path", DEFAULT_GITHUB_JSON)),
        )
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            era = st.number_input("Era", 0, 20, 0)
        with ic2:
            n_batch = st.number_input("Batch size", 8, 256, 32, 8)
        with ic3:
            n_steps = st.number_input("Train steps", 1, 200, 5)

        if st.button("Run train steps"):
            try:
                losses: list[float] = []
                use_file = src == "Local github_issues.json (via API)"
                gh_path = Path(st.session_state.get("github_json_path", DEFAULT_GITHUB_JSON))
                if use_file:
                    if not gh_path.is_file():
                        st.error(f"Missing {gh_path}")
                    else:
                        data = json.loads(gh_path.read_text(encoding="utf-8"))
                        pool = [
                            s
                            for s in data.get("samples", [])
                            if int(s.get("era", -1)) == int(era)
                        ]
                        key = f"gh_train_idx_{era}"
                        idx = int(st.session_state.get(key, 0))
                        with _client(base) as c:
                            for _ in range(int(n_steps)):
                                chunk = pool[idx : idx + int(n_batch)]
                                if len(chunk) < int(n_batch):
                                    break
                                inc = [
                                    {"text": s["text"], "label": int(s["label"])}
                                    for s in chunk
                                ]
                                tr = c.post(
                                    "/train",
                                    json={"era": int(era), "incidents": inc},
                                )
                                tr.raise_for_status()
                                losses.append(float(tr.json()["loss"]))
                                idx += int(n_batch)
                        st.session_state[key] = idx
                        if losses:
                            st.success(f"{len(losses)} steps · last loss {losses[-1]:.4f}")
                        _refresh_metrics_alert(base)
                else:
                    with _client(base) as c:
                        for _ in range(int(n_steps)):
                            sb = c.get(
                                "/demo/synthetic-batch",
                                params={"era": int(era), "n": int(n_batch)},
                            )
                            sb.raise_for_status()
                            inc = sb.json()["incidents"]
                            tr = c.post("/train", json={"era": int(era), "incidents": inc})
                            tr.raise_for_status()
                            losses.append(float(tr.json()["loss"]))
                    st.success(f"{n_steps} steps · last loss {losses[-1]:.4f}")
                    _refresh_metrics_alert(base)
            except Exception as e:
                st.error(str(e))

        if st.button("Close era (POST /era/close)"):
            try:
                with _client(base) as c:
                    r = c.post("/era/close", json={"era": int(era)})
                    r.raise_for_status()
                    st.json(r.json())
                _refresh_metrics_alert(base)
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("incremental_losses"):
            st.line_chart(st.session_state["incremental_losses"])

    with tab_analytics:
        st.subheader("Live metrics dashboard")
        if st.button("Refresh from API", key="an_refresh"):
            try:
                _refresh_metrics_alert(base)
            except Exception as e:
                st.error(str(e))
        m = st.session_state.get("metrics")
        al = st.session_state.get("alert")
        if m:
            _render_alert_card(al)
            _render_analytics_dashboard(m, al, chart_scope="analytics")
            with st.expander("Raw metrics JSON"):
                st.json(m)
        else:
            st.info("No metrics in session — run Home, Data pipeline, or Training lab first.")

    with tab_explore:
        st.markdown("##### Inference & retrieval")
        st.caption(
            "Text is encoded on the **server** (same encoder as training). "
            "Run **Training lab**, **Data pipeline → replay**, or **Home → simulation** first for meaningful results."
        )
        t1, t2, t3 = st.tabs(["Predict", "Similar", "Insight"])
        with t1:
            txt_p = st.text_area(
                "Incident text",
                height=110,
                key="pred_txt",
                placeholder="e.g. Prometheus scrape timeout on kube-state-metrics…",
            )
            if st.button("POST /predict (text)", type="primary", key="pred_txt_btn"):
                try:
                    with _client(base) as c:
                        r = c.post(
                            "/predict",
                            json={"incident": {"text": txt_p.strip()}},
                        )
                        r.raise_for_status()
                        _render_predict_summary(r.json())
                except Exception as e:
                    st.error(str(e))
            st.divider()
            raw_f = st.text_input(
                "Or comma-separated features",
                placeholder="0.1, 0.2, … (length = API feature_dim)",
                key="pred_f",
            )
            if st.button("POST /predict (features)", key="pred_feat_btn"):
                try:
                    vals = [float(x.strip()) for x in raw_f.split(",") if x.strip()]
                    with _client(base) as c:
                        r = c.post(
                            "/predict",
                            json={"incident": {"features": vals}},
                        )
                        r.raise_for_status()
                        _render_predict_summary(r.json())
                except Exception as e:
                    st.error(str(e))
        with t2:
            txt_s = st.text_area(
                "Query text",
                height=110,
                key="sim_txt",
                placeholder="Describe an incident to find nearest buffer neighbors…",
            )
            k_sim = st.slider("Neighbors k", 1, 20, 5, key="sim_k")
            if st.button("POST /similar (text)", type="primary", key="sim_btn"):
                try:
                    with _client(base) as c:
                        r = c.post(
                            "/similar",
                            json={
                                "incident": {"text": txt_s.strip()},
                                "k": int(k_sim),
                            },
                        )
                        r.raise_for_status()
                        _render_similar_table(r.json())
                except Exception as e:
                    st.error(str(e))
        with t3:
            txt_i = st.text_area(
                "Incident text",
                height=110,
                key="ins_txt",
                placeholder="Full incident description for class + neighbors + suggested fix…",
            )
            ik = st.slider("k neighbors", 1, 20, 5, key="ins_k")
            c_if, c_llm = st.columns(2)
            with c_if:
                inc_forget = st.checkbox("Include forgetting context", value=True, key="ins_forget")
            with c_llm:
                inc_llm = st.checkbox("Include LLM (needs API key on server)", value=False, key="ins_llm")
            if st.button("POST /predict/insight", type="primary", key="ins_btn"):
                try:
                    with _client(base) as c:
                        r = c.post(
                            "/predict/insight",
                            json={
                                "incident": {"text": txt_i.strip()},
                                "k_neighbors": int(ik),
                                "include_forgetting": inc_forget,
                                "include_llm": inc_llm,
                            },
                        )
                        r.raise_for_status()
                        _render_insight_panel(r.json())
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("Batch file upload")
        up = st.file_uploader("CSV or JSON", type=["csv", "json"])
        tc, lc = st.columns(2)
        with tc:
            tcol = st.text_input("Text column", "text")
        with lc:
            lcol = st.text_input("Label column", "label")
        era_b = st.number_input("Era", 0, 50, 0, key="up_era")
        ch = st.number_input("Chunk size", 8, 2048, 128, 8, key="up_ch")
        close_end = st.checkbox("Close era after ingest", value=False)
        if st.button("POST /ingest/batch") and up is not None:
            try:
                content = up.getvalue()
                files = {
                    "file": (
                        up.name,
                        content,
                        "text/csv"
                        if up.name.lower().endswith(".csv")
                        else "application/json",
                    )
                }
                form = {
                    "text_column": tcol,
                    "label_column": lcol,
                    "era": str(int(era_b)),
                    "default_incident_type": "",
                    "default_fix": "",
                    "memory_tier": "short_term",
                    "chunk_size": str(int(ch)),
                    "close_era_at_end": "true" if close_end else "false",
                }
                with _client(base, 600.0) as c:
                    r = c.post("/ingest/batch", files=files, data=form)
                    r.raise_for_status()
                    st.json(r.json())
                _refresh_metrics_alert(base)
            except Exception as e:
                st.error(str(e))


main()
