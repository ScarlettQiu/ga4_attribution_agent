#!/usr/bin/env python3
"""Entry point for the GA4 Attribution Agent.

Usage:
    python main.py                        # Interactive mode
    python main.py --project my-project   # Pre-fill project ID
    python main.py --sql-only             # Just show the generated SQL
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def check_dependencies(sql_only: bool = False) -> None:
    # sql_only mode only needs pandas + numpy (no anthropic/bigquery)
    required = ["pandas", "numpy"] if sql_only else ["anthropic", "google.cloud.bigquery", "pandas", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.split(".")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies. Run:\n")
        print("  pip install -r requirements.txt\n")
        print("Missing:", ", ".join(missing))
        sys.exit(1)


def main() -> None:
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="GA4 Attribution Agent — multi-touch attribution powered by Claude"
    )
    parser.add_argument("--project", help="GCP project ID (pre-fill to skip that question)")
    parser.add_argument("--dataset", help="BigQuery dataset ID")
    parser.add_argument(
        "--sql-only",
        action="store_true",
        help="Print the generated SQL template and exit",
    )
    args = parser.parse_args()

    if args.sql_only:
        # Print a sample SQL so users can review the query structure
        from ga4_attribution.sql_builder import build_journey_sql

        sql = build_journey_sql(
            project_id=args.project or "your-project",
            dataset_id=args.dataset or "analytics_123456789",
            start_date="20240101",
            end_date="20240131",
            conversion_events=["purchase"],
            lookback_days=30,
            channel_grouping="default",
        )
        print(sql)
        return

    # Validate API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Copy .env.example to .env and add your key, or set the env var directly.")
        sys.exit(1)

    from ga4_attribution.bigquery import BigQueryClient
    from ga4_attribution.agent import GA4AttributionAgent

    bq = BigQueryClient(project_id=args.project)
    agent = GA4AttributionAgent(bq_client=bq)

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
