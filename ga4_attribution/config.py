"""Company-specific configuration helpers for GA4 attribution."""

from __future__ import annotations

import csv
from pathlib import Path


CHANNEL_MAPPING_TEMPLATE = (
    "source,medium,campaign_contains,channel_label\n"
    "google,cpc,pmax,PMAX\n"
    "google,cpc,brand,Branded Search\n"
    "google,cpc,,Generic Paid Search\n"
    ",,email,Email\n"
)


def load_channel_mapping(csv_path: str) -> list[dict[str, str]]:
    """Parse a channel mapping CSV and return a list of rule dicts.

    Columns: source, medium, campaign_contains, channel_label
    Empty cells = wildcard. Rows are evaluated top-to-bottom; first match wins.
    Rows without a channel_label are skipped.
    """
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Channel mapping file not found: {path}")

    rules: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("channel_label", "").strip()
            if not label:
                continue
            rules.append({
                "source":            row.get("source", "").strip(),
                "medium":            row.get("medium", "").strip(),
                "campaign_contains": row.get("campaign_contains", "").strip(),
                "channel_label":     label,
            })
    return rules
