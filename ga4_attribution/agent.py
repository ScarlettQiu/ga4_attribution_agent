"""GA4 Attribution Agent — Claude-powered conversational orchestrator.

The agent uses Claude (claude-opus-4-6 with adaptive thinking) and tool use to:
  1. Discover the GA4 BigQuery dataset
  2. Ask the user key questions about their setup
  3. Generate + execute standardised SQL for journey extraction
  4. Run all attribution models
  5. Present results
"""

from __future__ import annotations

import json
import sys
from typing import Any

import anthropic

from .bigquery import BigQueryClient
from .attribution import run_all_models, AVAILABLE_MODELS
from .formatters import (
    print_attribution_table,
    print_journey_preview,
    print_model_explanations,
)
from .sql_builder import build_journey_sql


SYSTEM_PROMPT = """You are an expert GA4 digital marketing analyst specialising in multi-touch attribution.
Your job is to guide the user through a complete attribution analysis on their GA4 Google BigQuery data.

## Your workflow

### Phase 1 — Discovery
1. Ask for the BigQuery **project ID** and **dataset ID** (GA4 exports look like `analytics_<property_id>`).
2. Ask for the **date range** they want to analyse (YYYYMMDD format, e.g. 20240101 to 20240131).
3. Call `list_events` to show what events exist, then ask which event(s) should count as **conversions**
   (common ones: purchase, generate_lead, form_submit, sign_up, begin_checkout).
4. Ask for the **lookback window** in days (default 30) — how far back before a conversion to include touchpoints.
5. Ask how they want channels grouped: **"default"** (GA4-style channel groups like Organic Search, Paid Search,
   Email, etc.) or **"source_medium"** (raw google / cpc style strings).

### Phase 2 — Data validation
6. Call `list_channels` to show the top source/medium combos so the user can verify the data looks right.
7. Call `preview_journeys` to show a handful of example customer journeys.

### Phase 3 — Attribution analysis
8. Call `run_attribution` with the confirmed configuration to extract journey data and run all models.
9. Present results clearly, explain what the numbers mean, and highlight interesting differences between models.
   - If Shapley/Markov diverge from Last Touch significantly, explain what that reveals.
   - Suggest which model might be most appropriate for their business.

## GA4 BigQuery notes (so you give correct guidance)
- Tables are named `events_YYYYMMDD` — dates must be in YYYYMMDD format.
- `user_pseudo_id` is the anonymous GA4 user identifier.
- `traffic_source.source` / `.medium` / `.name` are the session-level channel fields.
- Revenue is typically in `event_params` under key `revenue` or `value` for purchase events.

## Tone
Be friendly, concise, and educational. Explain jargon. If the user gives you a date like "last month",
convert it to YYYYMMDD yourself based on today's date (2026-03-30). If a parameter is missing,
use sensible defaults (30-day lookback, default channel grouping) rather than blocking on it.
"""


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_events",
        "description": (
            "Query BigQuery to list all GA4 event names with counts and unique user counts "
            "for the specified project, dataset, and date range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":  {"type": "string", "description": "GCP project ID"},
                "dataset_id":  {"type": "string", "description": "BigQuery dataset ID (e.g. analytics_123456)"},
                "start_date":  {"type": "string", "description": "Start date YYYYMMDD"},
                "end_date":    {"type": "string", "description": "End date YYYYMMDD"},
            },
            "required": ["project_id", "dataset_id", "start_date", "end_date"],
        },
    },
    {
        "name": "list_channels",
        "description": (
            "Query BigQuery to list source/medium combinations with session counts, "
            "so the user can see which channels exist in their data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":  {"type": "string"},
                "dataset_id":  {"type": "string"},
                "start_date":  {"type": "string", "description": "YYYYMMDD"},
                "end_date":    {"type": "string", "description": "YYYYMMDD"},
            },
            "required": ["project_id", "dataset_id", "start_date", "end_date"],
        },
    },
    {
        "name": "preview_journeys",
        "description": (
            "Fetch a few example multi-touch customer journeys from BigQuery to validate "
            "that the data and channel grouping look correct before running the full analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":         {"type": "string"},
                "dataset_id":         {"type": "string"},
                "start_date":         {"type": "string", "description": "YYYYMMDD"},
                "end_date":           {"type": "string", "description": "YYYYMMDD"},
                "conversion_events":  {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of GA4 event names that count as conversions",
                },
                "lookback_days":      {"type": "integer", "default": 30},
                "channel_grouping":   {
                    "type": "string",
                    "enum": ["default", "source_medium"],
                    "default": "default",
                },
            },
            "required": ["project_id", "dataset_id", "start_date", "end_date", "conversion_events"],
        },
    },
    {
        "name": "run_attribution",
        "description": (
            "Execute the full attribution analysis: extract customer journeys from BigQuery "
            "and run all attribution models (Last Touch, First Touch, Linear, Time Decay, "
            "Position-Based, Shapley, Markov Chain). Returns a table of results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":         {"type": "string"},
                "dataset_id":         {"type": "string"},
                "start_date":         {"type": "string", "description": "YYYYMMDD"},
                "end_date":           {"type": "string", "description": "YYYYMMDD"},
                "conversion_events":  {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "lookback_days":      {"type": "integer", "default": 30},
                "channel_grouping":   {
                    "type": "string",
                    "enum": ["default", "source_medium"],
                    "default": "default",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Models to run. Defaults to all: {AVAILABLE_MODELS}",
                },
            },
            "required": ["project_id", "dataset_id", "start_date", "end_date", "conversion_events"],
        },
    },
    {
        "name": "show_sql",
        "description": "Show the SQL query that would be used for journey extraction, for transparency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":         {"type": "string"},
                "dataset_id":         {"type": "string"},
                "start_date":         {"type": "string"},
                "end_date":           {"type": "string"},
                "conversion_events":  {"type": "array", "items": {"type": "string"}},
                "lookback_days":      {"type": "integer", "default": 30},
                "channel_grouping":   {"type": "string", "default": "default"},
            },
            "required": ["project_id", "dataset_id", "start_date", "end_date", "conversion_events"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GA4AttributionAgent:
    """Runs an interactive CLI conversation powered by Claude with tool use."""

    def __init__(self, bq_client: BigQueryClient | None = None):
        self.client = anthropic.Anthropic()
        self.bq = bq_client or BigQueryClient()
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive attribution analysis session."""
        print("\n" + "=" * 70)
        print("  GA4 Attribution Agent")
        print("  Powered by Claude (claude-opus-4-6)")
        print("=" * 70)
        print_model_explanations()
        print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

        # Kick off the conversation
        self._assistant_turn("Hello! I'm ready to help you run a multi-touch attribution analysis on your GA4 BigQuery data. Let's start — what is your **Google Cloud project ID** and **BigQuery dataset ID** for your GA4 export?")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            self.messages.append({"role": "user", "content": user_input})
            self._call_claude()

    # ------------------------------------------------------------------
    # Claude interaction loop
    # ------------------------------------------------------------------

    def _call_claude(self) -> None:
        """Send messages to Claude and handle tool use in a loop."""
        while True:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
            )

            # Collect assistant content
            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            # Print text blocks
            for block in assistant_content:
                if block.type == "text" and block.text.strip():
                    self._assistant_turn(block.text)

            # If no tool calls, we're done
            if response.stop_reason == "end_turn":
                break

            # Handle tool use
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})
            else:
                break  # No tool calls despite stop_reason == "tool_use" — shouldn't happen

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> Any:
        """Dispatch a tool call and return JSON-serialisable result."""
        print(f"\n  ⚙  Running tool: {name} …")
        try:
            if name == "list_events":
                result = self.bq.list_events(**inputs)
                print(f"  ✓  Found {len(result)} event types")
                return result

            elif name == "list_channels":
                result = self.bq.list_channels(**inputs)
                print(f"  ✓  Found {len(result)} channel combinations")
                return result

            elif name == "preview_journeys":
                rows = self.bq.preview_journeys(**inputs)
                print_journey_preview(rows)
                return rows

            elif name == "run_attribution":
                return self._run_attribution_tool(inputs)

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
                print("\n--- Generated SQL ---")
                print(sql)
                print("--- End SQL ---\n")
                return {"sql": sql, "status": "displayed"}

            else:
                return {"error": f"Unknown tool: {name}"}

        except Exception as exc:
            error_msg = f"Tool '{name}' failed: {exc}"
            print(f"  ✗  {error_msg}", file=sys.stderr)
            return {"error": error_msg}

    def _run_attribution_tool(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the full attribution pipeline and display results."""
        print("  ⚙  Extracting customer journeys from BigQuery …")
        journeys_df = self.bq.extract_journeys(
            project_id=inputs["project_id"],
            dataset_id=inputs["dataset_id"],
            start_date=inputs["start_date"],
            end_date=inputs["end_date"],
            conversion_events=inputs["conversion_events"],
            lookback_days=inputs.get("lookback_days", 30),
            channel_grouping=inputs.get("channel_grouping", "default"),
        )

        n_journeys = journeys_df.groupby(
            ["user_pseudo_id", "conversion_timestamp"]
        ).ngroups if not journeys_df.empty else 0
        n_touchpoints = len(journeys_df)

        print(f"  ✓  {n_journeys:,} converting journeys, {n_touchpoints:,} touchpoints")

        if journeys_df.empty:
            return {
                "status": "no_data",
                "message": "No journeys found. Check your conversion events, date range, and dataset.",
            }

        models = inputs.get("models") or AVAILABLE_MODELS
        print(f"  ⚙  Running {len(models)} attribution models …")

        results_df = run_all_models(journeys_df, models=models)

        print_attribution_table(
            results_df,
            title=(
                f"Attribution Results — {inputs['start_date']} to {inputs['end_date']} "
                f"| {n_journeys:,} conversions"
            ),
        )

        # Stats summary for Claude
        total_value = journeys_df.groupby(
            ["user_pseudo_id", "conversion_timestamp"]
        )["conversion_value"].first().sum()

        return {
            "status": "success",
            "total_conversions": n_journeys,
            "total_conversion_value": round(float(total_value), 2),
            "n_touchpoints": n_touchpoints,
            "avg_path_length": round(journeys_df["total_touchpoints"].mean(), 2),
            "models_run": models,
            "results": results_df.to_dict(orient="records"),
            "top_channel_by_last_touch": (
                results_df.iloc[0]["channel"] if not results_df.empty and "last_touch" in results_df.columns
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _assistant_turn(text: str) -> None:
        print(f"\nAgent: {text}\n")
