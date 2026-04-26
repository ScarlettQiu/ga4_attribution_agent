#!/usr/bin/env python3
"""Standalone CLI for GA4 attribution analysis.

Called by the /ga4-attribution Claude Code skill:
    python run_attribution.py \
        --project my-project \
        --dataset analytics_123 \
        --start 20240101 \
        --end 20240131 \
        --events purchase \
        --lookback 30 \
        --grouping default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="GA4 Multi-Touch Attribution")
    parser.add_argument("--project",  required=True, help="GCP project ID")
    parser.add_argument("--dataset",  required=True, help="BigQuery dataset ID")
    parser.add_argument("--start",    required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end",      required=True, help="End date YYYYMMDD")
    parser.add_argument("--events",   required=True, nargs="+", help="Conversion event name(s)")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window in days")
    parser.add_argument("--grouping", default="default",
                        choices=["default", "source_medium"],
                        help="Channel grouping style")
    parser.add_argument("--models",       nargs="+", help="Attribution models to run (default: all)")
    parser.add_argument("--sql-only",     action="store_true", help="Print SQL and exit")
    parser.add_argument("--use-user-id",  action="store_true",
                        help="Use COALESCE(user_id, user_pseudo_id) for cross-device stitching")
    parser.add_argument("--channel-mapping", metavar="CSV_PATH",
                        help="Path to custom channel mapping CSV for UTM remapping")
    args = parser.parse_args()

    # ── Resolve custom channel mapping ───────────────────────────────────
    custom_channel_rules = None
    if args.channel_mapping:
        from ga4_attribution.config import load_channel_mapping
        custom_channel_rules = load_channel_mapping(args.channel_mapping)
        print(f"📋 Loaded {len(custom_channel_rules)} custom channel mapping rules")

    # ── SQL only mode ────────────────────────────────────────────────────
    if args.sql_only:
        from ga4_attribution.sql_builder import build_journey_sql
        sql = build_journey_sql(
            project_id=args.project,
            dataset_id=args.dataset,
            start_date=args.start,
            end_date=args.end,
            conversion_events=args.events,
            lookback_days=args.lookback,
            channel_grouping=args.grouping,
            use_user_id=args.use_user_id,
            custom_channel_rules=custom_channel_rules,
        )
        print(sql)
        return

    # ── Connect to BigQuery ──────────────────────────────────────────────
    print(f"\n🔌 Connecting to BigQuery project: {args.project}")
    try:
        from ga4_attribution.bigquery import BigQueryClient
        bq = BigQueryClient(project_id=args.project)
    except Exception as e:
        print(f"❌ BigQuery connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Extract journeys ─────────────────────────────────────────────────
    print(f"📥 Extracting journeys: {args.start} → {args.end}")
    print(f"   Conversion events : {', '.join(args.events)}")
    print(f"   Lookback window   : {args.lookback} days")
    print(f"   Channel grouping  : {args.grouping}")
    if args.use_user_id:
        print(f"   User key          : user_id (cross-device stitching enabled)")
    if args.channel_mapping:
        print(f"   Channel mapping   : {args.channel_mapping}")

    try:
        journeys_df = bq.extract_journeys(
            project_id=args.project,
            dataset_id=args.dataset,
            start_date=args.start,
            end_date=args.end,
            conversion_events=args.events,
            lookback_days=args.lookback,
            channel_grouping=args.grouping,
            use_user_id=args.use_user_id,
            custom_channel_rules=custom_channel_rules,
        )
    except Exception as e:
        print(f"❌ Query failed: {e}", file=sys.stderr)
        sys.exit(1)

    if journeys_df.empty:
        print("\n⚠️  No journey data found.")
        print("   Check your conversion events, date range, and dataset name.")
        sys.exit(0)

    # ── Stats ────────────────────────────────────────────────────────────
    import pandas as pd
    n_journeys = journeys_df.groupby(["user_pseudo_id", "conversion_timestamp"]).ngroups
    total_value = journeys_df.groupby(
        ["user_pseudo_id", "conversion_timestamp"]
    )["conversion_value"].first().sum()
    avg_path = journeys_df["total_touchpoints"].mean()

    print(f"\n   User key              : {'user_id (cross-device)' if args.use_user_id else 'user_pseudo_id'}")
    print(f"\n✅ Found {n_journeys:,} converting journeys")
    print(f"   Total conversion value : ${total_value:,.2f}")
    print(f"   Avg path length        : {avg_path:.1f} touchpoints")

    # ── Run attribution ──────────────────────────────────────────────────
    from ga4_attribution.attribution import run_all_models, AVAILABLE_MODELS
    models = args.models or AVAILABLE_MODELS
    print(f"\n⚙️  Running {len(models)} attribution model(s): {', '.join(models)}\n")

    results_df = run_all_models(journeys_df, models=models)

    # ── Print table ──────────────────────────────────────────────────────
    try:
        from tabulate import tabulate
        table = tabulate(
            results_df,
            headers="keys",
            tablefmt="rounded_outline",
            floatfmt=",.1f",
            showindex=False,
        )
    except ImportError:
        table = results_df.to_string(index=False, float_format=lambda x: f"{x:,.1f}")

    print("=" * 70)
    print(f"  Attribution Results — {args.start} → {args.end}")
    print("=" * 70)
    print(table)
    print()

    # ── Model comparison insight ─────────────────────────────────────────
    if "last_touch" in results_df.columns and "shapley" in results_df.columns:
        top_last = results_df.iloc[0]["channel"]
        top_shapley = results_df.sort_values("shapley", ascending=False).iloc[0]["channel"]
        if top_last != top_shapley:
            print(f"💡 Note: Last Touch credits '{top_last}' most, but Shapley")
            print(f"   credits '{top_shapley}' — worth investigating the difference.")
        print()


if __name__ == "__main__":
    main()
