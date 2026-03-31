"""Result formatting and display helpers."""

from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd


def print_attribution_table(df: pd.DataFrame, title: str = "Attribution Results") -> None:
    """Print a nicely formatted attribution results table."""
    try:
        from tabulate import tabulate  # noqa: PLC0415
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    if df.empty:
        print("  (no data)")
        return

    if use_tabulate:
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", floatfmt=".1f", showindex=False))
    else:
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    print()


def print_journey_preview(rows: list[dict[str, Any]]) -> None:
    """Pretty-print a few example journeys."""
    if not rows:
        print("  (no journey data)")
        return

    # Group by user + conversion
    by_journey: dict[str, list] = {}
    for row in rows:
        key = f"{row.get('user_pseudo_id','?')[:8]}…|{row.get('conversion_timestamp','?')[:19]}"
        by_journey.setdefault(key, []).append(row)

    print(f"\n{'─' * 60}")
    print("  Example Customer Journeys")
    print(f"{'─' * 60}")
    for journey_key, touchpoints in list(by_journey.items())[:5]:
        value = touchpoints[0].get("conversion_value", 0)
        print(f"\n  User: {journey_key}   Value: {value:.2f}")
        for tp in sorted(touchpoints, key=lambda x: x.get("touchpoint_position", 0)):
            pos = tp.get("touchpoint_position", "?")
            total = tp.get("total_touchpoints", "?")
            channel = tp.get("channel", "?")
            ts = str(tp.get("session_timestamp", ""))[:19]
            print(f"    [{pos}/{total}]  {ts}  →  {channel}")
    print()


def attribution_to_json(df: pd.DataFrame) -> str:
    """Serialize attribution results to JSON string."""
    return json.dumps(df.to_dict(orient="records"), indent=2)


def print_model_explanations() -> None:
    """Print one-line explanations of each attribution model."""
    models = {
        "last_touch":     "100% credit to the final touchpoint before conversion.",
        "first_touch":    "100% credit to the first touchpoint that started the journey.",
        "linear":         "Equal credit split across all touchpoints.",
        "time_decay":     "Exponential decay — recent touchpoints get more credit.",
        "position_based": "U-shaped: 40% first, 40% last, 20% spread across middle.",
        "shapley":        "Game-theory: marginal contribution of each channel to the conversion.",
        "markov":         "Markov Chain removal effect — how much conversions drop if a channel is removed.",
    }
    print("\n📊 Attribution Models:")
    for name, desc in models.items():
        print(f"  {name:<18}  {desc}")
    print()
