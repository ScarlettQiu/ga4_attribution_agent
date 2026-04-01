"""GA4 Attribution Agent — Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Load .env if present (local development)
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from ga4_attribution.agent import SYSTEM_PROMPT, TOOLS
from ga4_attribution.bigquery import BigQueryClient
from ga4_attribution.streamlit_tools import execute_tool


def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first (Streamlit Cloud), then env var (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


@st.cache_resource
def _build_gcp_credentials():
    """Build GCP credentials from st.secrets[gcp_service_account] if present."""
    try:
        sa = st.secrets.get("gcp_service_account")
        if sa and sa.get("private_key"):
            from google.oauth2 import service_account  # type: ignore[import-untyped]
            return service_account.Credentials.from_service_account_info(dict(sa))
    except Exception:
        pass
    return None  # Fall back to Application Default Credentials

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GA4 Attribution Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "last_touch":     "#E63946",
    "first_touch":    "#457B9D",
    "linear":         "#2A9D8F",
    "time_decay":     "#E9C46A",
    "position_based": "#9B5DE5",
    "shapley":        "#F4A261",
    "markov":         "#06D6A0",
}

TOOL_LABELS = {
    "list_events":      "Querying GA4 event names…",
    "list_channels":    "Querying source / medium combinations…",
    "preview_journeys": "Fetching example customer journeys…",
    "run_attribution":  "Running 7 attribution models…",
    "show_sql":         "Generating attribution SQL…",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages": [],           # Anthropic API message history
        "chat_display": [],       # Display-only history [{role, text, tools, artifact}]
        "attribution_results": None,
        "bq_client": None,
        "bq_project": None,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# BigQuery client (cached by project_id)
# ---------------------------------------------------------------------------

def get_bq_client(project_id: str | None = None) -> BigQueryClient:
    if (
        st.session_state.bq_client is None
        or st.session_state.bq_project != project_id
    ):
        credentials = _build_gcp_credentials()
        st.session_state.bq_client = BigQueryClient(
            project_id=project_id or None,
            credentials=credentials,
        )
        st.session_state.bq_project = project_id
    return st.session_state.bq_client

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str | None:
    """Render sidebar config. Returns a pre-filled message string if the
    user clicks 'Start analysis', otherwise None."""

    with st.sidebar:
        st.title("⚙️ Quick Setup")
        st.caption("Fill in your GA4 details to auto-start the conversation.")

        project_id = st.text_input(
            "GCP Project ID",
            placeholder="my-gcp-project",
            key="sb_project",
        )
        dataset_id = st.text_input(
            "BigQuery Dataset ID",
            placeholder="analytics_123456789",
            key="sb_dataset",
        )

        today = date.today()
        default_start = today.replace(day=1) - timedelta(days=1)
        default_start = default_start.replace(day=1)  # first day of last month

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=default_start, key="sb_start")
        with col2:
            end_date = st.date_input("End date", value=today - timedelta(days=1), key="sb_end")

        conversion_event = st.text_input(
            "Conversion event",
            placeholder="purchase",
            value="purchase",
            key="sb_event",
        )
        lookback = st.slider("Lookback window (days)", 7, 90, 30, key="sb_lookback")
        channel_grouping = st.selectbox(
            "Channel grouping",
            ["default (GA4 groups)", "source / medium"],
            key="sb_grouping",
        )

        st.divider()
        injected_msg = None

        if st.button("▶ Start analysis", type="primary", use_container_width=True):
            if project_id and dataset_id:
                grouping_val = "default" if "default" in channel_grouping else "source_medium"
                injected_msg = (
                    f"Please analyse my GA4 data. "
                    f"Project: **{project_id}**, dataset: **{dataset_id}**, "
                    f"dates: **{start_date.strftime('%Y%m%d')}** to **{end_date.strftime('%Y%m%d')}**, "
                    f"conversion event: **{conversion_event}**, "
                    f"lookback: **{lookback} days**, "
                    f"channel grouping: **{grouping_val}**."
                )
            else:
                st.warning("Enter a project ID and dataset ID first.")

        st.divider()
        st.markdown("**Attribution models**")
        for model, color in MODEL_COLORS.items():
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};margin-right:6px"></span>'
                f'`{model}`',
                unsafe_allow_html=True,
            )

        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_display = []
            st.session_state.attribution_results = None
            st.session_state.running = False
            st.rerun()

        if st.session_state.get("running"):
            if st.button("⚠️ Unlock input", use_container_width=True, type="secondary"):
                st.session_state.running = False
                st.rerun()

    return injected_msg

# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def render_attribution_artifact(artifact: dict[str, Any]) -> None:
    """Render the attribution results table + chart + SQL."""
    df: pd.DataFrame = artifact["results_df"]
    sql: str = artifact["sql"]
    meta: dict = artifact["meta"]

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversions", f"{meta['total_conversions']:,}")
    c2.metric("Total value", f"${meta['total_conversion_value']:,.2f}")
    c3.metric("Avg path length", f"{meta['avg_path_length']:.1f} touches")

    st.markdown(
        f"_Analysis period: {meta['start_date']} → {meta['end_date']}_"
    )

    # Table
    with st.expander("📋 Attribution table", expanded=True):
        display_df = df.set_index("channel")
        # Format all numeric columns as 2 decimal places
        styled = display_df.style.format("{:,.1f}").highlight_max(
            axis=0, color="#d4f5e9"
        )
        st.dataframe(styled, use_container_width=True)

    # Chart — grouped bar
    with st.expander("📊 Chart: models compared", expanded=True):
        fig = _build_chart(df, meta)
        st.plotly_chart(fig, use_container_width=True)

    # SQL
    with st.expander("🔍 Generated SQL", expanded=False):
        st.code(sql, language="sql")


def render_journey_preview(rows: list[dict[str, Any]]) -> None:
    """Render a compact journey preview table."""
    if not rows:
        st.caption("No journey data.")
        return
    preview_df = pd.DataFrame(rows)[
        [c for c in
         ["user_pseudo_id", "session_timestamp", "channel", "touchpoint_position",
          "total_touchpoints", "conversion_value"]
         if c in pd.DataFrame(rows).columns]
    ]
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


def _build_chart(df: pd.DataFrame, meta: dict) -> go.Figure:
    channels = df["channel"].tolist()
    model_cols = [c for c in df.columns if c != "channel"]

    # Horizontal layout when many channels
    horizontal = len(channels) > 7

    fig = go.Figure()
    for model in model_cols:
        color = MODEL_COLORS.get(model, "#888888")
        if horizontal:
            fig.add_trace(go.Bar(
                name=model,
                y=channels,
                x=df[model].tolist(),
                orientation="h",
                marker_color=color,
            ))
        else:
            fig.add_trace(go.Bar(
                name=model,
                x=channels,
                y=df[model].tolist(),
                marker_color=color,
            ))

    fig.update_layout(
        barmode="group",
        title=f"Attribution by channel — {meta['start_date']} to {meta['end_date']}",
        height=420 if not horizontal else max(420, len(channels) * 55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e5e5") if not horizontal else dict(),
        yaxis=dict(gridcolor="#e5e5e5") if not horizontal else dict(),
    )
    return fig

# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    for entry in st.session_state.chat_display:
        role = entry["role"]
        with st.chat_message(role):
            if entry.get("text"):
                st.markdown(entry["text"])
            # Re-render any stored artifacts
            artifact = entry.get("artifact")
            if artifact:
                if artifact["type"] == "attribution":
                    render_attribution_artifact(artifact)
                elif artifact["type"] == "journey_preview":
                    render_journey_preview(artifact["rows"])
                elif artifact["type"] == "sql":
                    with st.expander("Generated SQL", expanded=False):
                        st.code(artifact["sql"], language="sql")

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_loop(user_message: str) -> None:
    """Run the Claude tool-use loop for one user turn, streaming live."""

    # Append user turn
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.chat_display.append({"role": "user", "text": user_message})

    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_message)

    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        with st.chat_message("assistant"):
            st.error(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file (local) or Streamlit Cloud Secrets panel."
            )
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Infer project_id from messages history for BQ client
    project_id = _extract_project_id()
    bq = get_bq_client(project_id)

    # Tool-use loop (same structure as agent.py._call_claude)
    while True:
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            accumulated_text = ""
            tool_calls_raw: list[dict] = []
            current_tool: dict | None = None
            assistant_artifact = None

            # ── Streaming ────────────────────────────────────────────
            try:
                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=8192,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=st.session_state.messages,
                ) as stream:
                    for event in stream:
                        etype = event.type

                        if etype == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                current_tool = {
                                    "id": block.id,
                                    "name": block.name,
                                    "input_str": "",
                                }
                        elif etype == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                accumulated_text += delta.text
                                text_placeholder.markdown(accumulated_text + "▌")
                            elif delta.type == "input_json_delta" and current_tool:
                                current_tool["input_str"] += delta.partial_json
                        elif etype == "content_block_stop":
                            if current_tool:
                                tool_calls_raw.append(current_tool)
                                current_tool = None

                    final_msg = stream.get_final_message()

            except anthropic.APIConnectionError as e:
                st.error(f"Connection error: {e}")
                return
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
                return
            except anthropic.APIStatusError as e:
                st.error(f"API error {e.status_code}: {e.message}")
                return

            # Finalise text (remove cursor)
            if accumulated_text:
                text_placeholder.markdown(accumulated_text)
            else:
                text_placeholder.empty()

            # Append assistant message to API history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_msg.content,
            })

            # ── Execute tool calls ────────────────────────────────────
            tool_results = []
            for tc in tool_calls_raw:
                name = tc["name"]
                try:
                    inputs = json.loads(tc["input_str"]) if tc["input_str"] else {}
                except json.JSONDecodeError:
                    inputs = {}

                label = TOOL_LABELS.get(name, f"Running {name}…")

                with st.status(label, expanded=False) as status:
                    claude_result, ui_artifact = execute_tool(name, inputs, bq)

                    if isinstance(claude_result, dict) and "error" in claude_result:
                        status.update(label=f"❌ {label}", state="error")
                        st.error(claude_result["error"])
                    else:
                        status.update(
                            label=label.replace("…", " ✓"),
                            state="complete",
                            expanded=False,
                        )

                    # Render artifact inline
                    if ui_artifact:
                        artifact_type = ui_artifact.get("type")
                        if artifact_type == "attribution":
                            render_attribution_artifact(ui_artifact)
                            st.session_state.attribution_results = ui_artifact
                            assistant_artifact = ui_artifact
                        elif artifact_type == "journey_preview":
                            render_journey_preview(ui_artifact["rows"])
                            assistant_artifact = ui_artifact
                        elif artifact_type == "sql":
                            with st.expander("Generated SQL", expanded=False):
                                st.code(ui_artifact["sql"], language="sql")
                            assistant_artifact = ui_artifact

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(claude_result)
                    if not isinstance(claude_result, str)
                    else claude_result,
                })

            # Save this assistant turn to display history
            st.session_state.chat_display.append({
                "role": "assistant",
                "text": accumulated_text or None,
                "artifact": assistant_artifact,
            })

            # If done, break
            if final_msg.stop_reason == "end_turn" or not tool_results:
                break

            # Append tool results and loop
            st.session_state.messages.append({
                "role": "user",
                "content": tool_results,
            })


def _extract_project_id() -> str | None:
    """Try to find a GCP project ID mentioned in conversation so far."""
    for msg in reversed(st.session_state.messages):
        content = msg.get("content", "")
        if isinstance(content, str) and "project" in content.lower():
            # Very naive: look for something that looks like a project ID
            import re
            m = re.search(r'project[:\s]+([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
            if m:
                return m.group(1)
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📊 GA4 Attribution Agent")
    st.caption(
        "Powered by Claude claude-opus-4-6 · "
        "Connects to Google BigQuery · "
        "7 attribution models"
    )

    # ── Config checks — show visible warnings at the top ──────────────────
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not set.** "
            "Add it to your `.env` file (local) or the Streamlit Cloud Secrets panel.",
            icon="🔑",
        )

    # Sidebar — may return a pre-filled message
    injected_msg = render_sidebar()

    # Render existing conversation
    render_chat_history()

    # Handle sidebar "Start analysis" button
    if injected_msg and not st.session_state.running:
        st.session_state.running = True
        try:
            run_agent_loop(injected_msg)
        finally:
            st.session_state.running = False
        st.rerun()

    # Chat input
    user_input = st.chat_input(
        "Ask me anything about your GA4 data…",
        disabled=st.session_state.running,
    )

    if user_input and not st.session_state.running:
        st.session_state.running = True
        try:
            run_agent_loop(user_input)
        finally:
            st.session_state.running = False
        st.rerun()

    # Welcome message on first load
    if not st.session_state.chat_display:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 Hi! I'm your GA4 attribution analyst.\n\n"
                "**To get started:**\n"
                "1. Fill in your BigQuery project and dataset in the sidebar, or\n"
                "2. Just tell me your project ID and dataset here in chat.\n\n"
                "I'll explore your data, then run **Last Touch, First Touch, Linear, "
                "Time Decay, Position-Based, Shapley, and Markov Chain** attribution "
                "models side by side."
            )


if __name__ == "__main__":
    main()
