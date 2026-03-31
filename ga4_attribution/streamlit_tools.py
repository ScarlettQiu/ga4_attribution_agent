"""Tool dispatch layer for the Streamlit UI.

Mirrors agent.py's _execute_tool but returns (claude_result, ui_artifact)
so the UI can render rich charts/tables without re-running queries.
"""

from __future__ import annotations

import json
from typing import Any

from .bigquery import BigQueryClient
from .attribution import run_all_models, AVAILABLE_MODELS
from .sql_builder import build_journey_sql


def execute_tool(
    name: str,
    inputs: dict[str, Any],
    bq_client: BigQueryClient,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute a tool call and return (claude_result, ui_artifact).

    claude_result  — JSON-serializable dict sent back in tool_result to Claude.
    ui_artifact    — None for most tools; rich dict for run_attribution / show_sql.
    """
    try:
        if name == "list_events":
            result = bq_client.list_events(**inputs)
            return result, None

        elif name == "list_channels":
            result = bq_client.list_channels(**inputs)
            return result, None

        elif name == "preview_journeys":
            rows = bq_client.preview_journeys(**inputs)
            artifact = {"type": "journey_preview", "rows": rows}
            return rows, artifact

        elif name == "show_sql":
            sql = build_journey_sql(
                project_id=inputs["project_id"],
                dataset_id=inputs["dataset_id"],
                start_date=inputs["start_date"],
                end_date=inputs["end_date"],
                conversion_events=inputs["conversion_events"],
                lookback_days=inputs.get("lookback_days", 30),
                channel_grouping=inputs.get("channel_grouping", "default"),
            )
            artifact = {"type": "sql", "sql": sql}
            return {"sql": sql, "status": "displayed"}, artifact

        elif name == "run_attribution":
            return _run_attribution(inputs, bq_client)

        else:
            return {"error": f"Unknown tool: {name}"}, None

    except Exception as exc:
        return {"error": str(exc)}, None


# ---------------------------------------------------------------------------
# Attribution pipeline
# ---------------------------------------------------------------------------

def _run_attribution(
    inputs: dict[str, Any],
    bq_client: BigQueryClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    journeys_df = bq_client.extract_journeys(
        project_id=inputs["project_id"],
        dataset_id=inputs["dataset_id"],
        start_date=inputs["start_date"],
        end_date=inputs["end_date"],
        conversion_events=inputs["conversion_events"],
        lookback_days=inputs.get("lookback_days", 30),
        channel_grouping=inputs.get("channel_grouping", "default"),
    )

    if journeys_df.empty:
        msg = "No journey data found. Check your conversion events, date range, and dataset."
        return {"status": "no_data", "message": msg}, None

    models = inputs.get("models") or AVAILABLE_MODELS
    results_df = run_all_models(journeys_df, models=models)

    # Metadata for the UI
    n_journeys = journeys_df.groupby(
        ["user_pseudo_id", "conversion_timestamp"]
    ).ngroups
    total_value = float(
        journeys_df.groupby(["user_pseudo_id", "conversion_timestamp"])
        ["conversion_value"].first().sum()
    )
    avg_path = float(journeys_df["total_touchpoints"].mean())

    # Build the SQL that was used
    sql = build_journey_sql(
        project_id=inputs["project_id"],
        dataset_id=inputs["dataset_id"],
        start_date=inputs["start_date"],
        end_date=inputs["end_date"],
        conversion_events=inputs["conversion_events"],
        lookback_days=inputs.get("lookback_days", 30),
        channel_grouping=inputs.get("channel_grouping", "default"),
    )

    ui_artifact = {
        "type": "attribution",
        "results_df": results_df,
        "sql": sql,
        "meta": {
            "total_conversions": n_journeys,
            "total_conversion_value": round(total_value, 2),
            "avg_path_length": round(avg_path, 2),
            "start_date": inputs["start_date"],
            "end_date": inputs["end_date"],
            "models_run": models,
        },
    }

    # Claude-compatible summary (no DataFrame)
    claude_result = {
        "status": "success",
        "total_conversions": n_journeys,
        "total_conversion_value": round(total_value, 2),
        "avg_path_length": round(avg_path, 2),
        "models_run": models,
        "results": results_df.to_dict(orient="records"),
    }

    return claude_result, ui_artifact
