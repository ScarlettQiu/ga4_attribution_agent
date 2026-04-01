"""GA4 Attribution Agent — LangChain SQL Agent version.

Uses LangChain's SQL agent with BigQuery so Claude can write and iterate
on GA4 SQL freely, plus a custom attribution tool that runs our Python models.

Run with:
    python langchain_agent.py
    python langchain_agent.py --project my-project --dataset analytics_123
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# GA4 schema primer — injected into the agent's system prompt so it writes
# correct BigQuery SQL for GA4's nested/repeated structure
# ---------------------------------------------------------------------------

GA4_SCHEMA_PRIMER = dedent("""
You are an expert GA4 BigQuery analyst. Key schema facts you MUST follow:

## Table structure
- Tables are named events_YYYYMMDD (e.g. events_20240115)
- Always filter with: WHERE _TABLE_SUFFIX BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'
- Full table reference: `{project}.{dataset}.events_*`

## Critical fields
- user_pseudo_id          — anonymous user identifier (TEXT)
- event_name              — name of the event (TEXT)
- event_timestamp         — microseconds since epoch (INTEGER)
- traffic_source.source   — referring source (TEXT)
- traffic_source.medium   — traffic medium (TEXT)
- traffic_source.name     — campaign name (TEXT)

## event_params — REPEATED RECORD (most important!)
To extract a parameter value, always use UNNEST:
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location')
  (SELECT value.int_value    FROM UNNEST(event_params) WHERE key = 'ga_session_id')
  (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'value')
  (SELECT value.float_value  FROM UNNEST(event_params) WHERE key = 'revenue')

Common event_params keys:
  ga_session_id (int)   — session identifier
  page_location (str)   — full page URL
  page_title (str)      — page title
  value (double/float)  — monetary value
  currency (str)        — currency code
  transaction_id (str)  — purchase transaction ID

## Common events
  session_start   — one per session, has traffic_source data
  page_view       — page viewed
  purchase        — completed purchase (has value, transaction_id params)
  add_to_cart     — product added to cart
  begin_checkout  — checkout started
  generate_lead   — lead form submitted
  sign_up         — user registration

## Revenue extraction for purchase events
  COALESCE(
    (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'revenue'),
    (SELECT value.float_value  FROM UNNEST(event_params) WHERE key = 'revenue'),
    (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'value'),
    1.0
  ) AS revenue

## Session-level channel data
Channel info lives on session_start events via traffic_source.*
To get channel per session:
  SELECT
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    traffic_source.source,
    traffic_source.medium,
    traffic_source.name AS campaign,
    MIN(event_timestamp) AS session_start_us
  FROM `project.dataset.events_*`
  WHERE _TABLE_SUFFIX BETWEEN 'start' AND 'end'
    AND event_name = 'session_start'
  GROUP BY 1, 2, 3, 4, 5
""").strip()


# ---------------------------------------------------------------------------
# Custom attribution tool
# ---------------------------------------------------------------------------

def _make_attribution_tool(project_id: str, dataset_id: str) -> Any:
    """Return a LangChain Tool that runs our Python attribution models."""
    from langchain_core.tools import tool  # noqa: PLC0415

    @tool
    def run_attribution_models(
        start_date: str,
        end_date: str,
        conversion_events: str,
        lookback_days: int = 30,
        channel_grouping: str = "default",
    ) -> str:
        """Run all 7 attribution models (Last Touch, First Touch, Linear,
        Time Decay, Position-Based, Shapley, Markov Chain) on GA4 journey data.

        Args:
            start_date: Start date in YYYYMMDD format (e.g. '20240101')
            end_date: End date in YYYYMMDD format (e.g. '20240131')
            conversion_events: Comma-separated conversion event names (e.g. 'purchase' or 'purchase,generate_lead')
            lookback_days: Days before conversion to include touchpoints (default 30)
            channel_grouping: 'default' for GA4 channel groups or 'source_medium' for raw source/medium
        """
        from ga4_attribution.bigquery import BigQueryClient  # noqa: PLC0415
        from ga4_attribution.attribution import run_all_models  # noqa: PLC0415

        events = [e.strip() for e in conversion_events.split(",")]

        try:
            bq = BigQueryClient(project_id=project_id)
            journeys_df = bq.extract_journeys(
                project_id=project_id,
                dataset_id=dataset_id,
                start_date=start_date,
                end_date=end_date,
                conversion_events=events,
                lookback_days=lookback_days,
                channel_grouping=channel_grouping,
            )
        except Exception as e:
            return f"Error extracting journeys: {e}"

        if journeys_df.empty:
            return "No journey data found. Check conversion events, date range, and dataset."

        n_journeys = journeys_df.groupby(
            ["user_pseudo_id", "conversion_timestamp"]
        ).ngroups
        total_value = float(
            journeys_df.groupby(["user_pseudo_id", "conversion_timestamp"])
            ["conversion_value"].first().sum()
        )
        avg_path = float(journeys_df["total_touchpoints"].mean())

        results_df = run_all_models(journeys_df)

        try:
            from tabulate import tabulate  # noqa: PLC0415
            table = tabulate(
                results_df,
                headers="keys",
                tablefmt="rounded_outline",
                floatfmt=",.1f",
                showindex=False,
            )
        except ImportError:
            table = results_df.to_string(index=False)

        return (
            f"✅ Attribution complete\n"
            f"   Journeys      : {n_journeys:,}\n"
            f"   Total value   : ${total_value:,.2f}\n"
            f"   Avg path len  : {avg_path:.1f} touches\n\n"
            f"{table}"
        )

    return run_attribution_models


# ---------------------------------------------------------------------------
# Build the LangChain SQL agent
# ---------------------------------------------------------------------------

def build_agent(project_id: str, dataset_id: str) -> Any:
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
    from langchain_community.utilities import SQLDatabase  # noqa: PLC0415
    from langchain_community.agent_toolkits import SQLDatabaseToolkit  # noqa: PLC0415
    from langchain.agents import create_react_agent, AgentExecutor  # noqa: PLC0415
    from langchain_core.prompts import PromptTemplate  # noqa: PLC0415

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # ── LLM ──────────────────────────────────────────��───────────────────
    llm = ChatAnthropic(
        model="claude-opus-4-6",
        api_key=api_key,
        max_tokens=8192,
    )

    # ── BigQuery SQLDatabase ──────────────────────────────────────────────
    print(f"🔌 Connecting to BigQuery: {project_id}.{dataset_id} …")
    try:
        db = SQLDatabase.from_uri(
            f"bigquery://{project_id}/{dataset_id}",
            # Limit schema inspection to avoid loading all event tables
            sample_rows_in_table_info=2,
            include_tables=None,
        )
    except Exception as e:
        print(f"❌ BigQuery connection failed: {e}")
        print("   Make sure you've run: gcloud auth application-default login")
        sys.exit(1)

    print("✅ Connected\n")

    # ── SQL toolkit tools ─────────────────────────────────────────────────
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()

    # ── Custom attribution tool ───────────────────────────────────────────
    attribution_tool = _make_attribution_tool(project_id, dataset_id)

    all_tools = sql_tools + [attribution_tool]

    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = dedent(f"""
        You are a GA4 digital marketing analyst with deep BigQuery SQL expertise.
        You help users explore their GA4 data and run multi-touch attribution analysis.

        {GA4_SCHEMA_PRIMER}

        ## Your capabilities
        1. **Explore** — use SQL tools to inspect tables, list events, show channels,
           preview data, answer ad-hoc questions about the GA4 dataset.
        2. **Attribute** — use the `run_attribution_models` tool to run all 7 attribution
           models (Last Touch, First Touch, Linear, Time Decay, Position-Based,
           Shapley, Markov Chain) and compare results.

        ## How to use the SQL tools
        - `sql_db_list_tables` — list available tables
        - `sql_db_schema` — get schema for specific tables
        - `sql_db_query` — run a SQL query (always use _TABLE_SUFFIX for date filtering)
        - `sql_db_query_checker` — validate SQL before running

        ## Current connection
        Project : {project_id}
        Dataset : {dataset_id}
        Full ref: `{project_id}.{dataset_id}.events_*`

        Always filter tables with `_TABLE_SUFFIX BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'`.
        Always use UNNEST() to access event_params values.

        {{tools}}

        Use this format:
        Question: the input question
        Thought: think about what to do
        Action: the action to take (one of [{{tool_names}}])
        Action Input: the input to the action
        Observation: the result
        ... (repeat Thought/Action/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the final answer to the question
    """).strip()

    prompt = PromptTemplate.from_template(
        system_prompt + "\n\nQuestion: {input}\n{agent_scratchpad}"
    )

    agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )

    return executor


# ---------------------------------------------------------------------------
# CLI chat loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GA4 Attribution — LangChain SQL Agent")
    parser.add_argument("--project", help="GCP project ID")
    parser.add_argument("--dataset", help="BigQuery dataset ID")
    args = parser.parse_args()

    project_id = args.project or os.environ.get("GCP_PROJECT_ID") or _ask("GCP project ID: ")
    dataset_id = args.dataset or _ask("BigQuery dataset ID (e.g. analytics_123456789): ")

    agent = build_agent(project_id, dataset_id)

    print("=" * 65)
    print("  GA4 Attribution Agent — LangChain SQL mode")
    print("=" * 65)
    print("Ask anything about your GA4 data. Examples:")
    print('  "What conversion events are available last month?"')
    print('  "Show me the top 10 source/medium combinations by sessions"')
    print('  "Run attribution analysis for purchase events in January 2024"')
    print('  "How many multi-touch journeys had 3+ touchpoints?"')
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = agent.invoke({"input": question})
            print(f"\nAgent: {result['output']}\n")
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)


if __name__ == "__main__":
    main()
