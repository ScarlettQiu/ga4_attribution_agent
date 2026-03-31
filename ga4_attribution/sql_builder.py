"""Standardized SQL query builder for GA4 BigQuery attribution analysis.

GA4 BigQuery schema notes:
- Tables: events_YYYYMMDD (daily) or events_intraday_YYYYMMDD
- _TABLE_SUFFIX is used for date-range filtering
- event_params is a REPEATED RECORD: [{key, value: {string_value, int_value, float_value, double_value}}]
- traffic_source: {source, medium, name} available on session_start events
- user_pseudo_id is the anonymous user key
- For revenue: event_params key='value' (double) or key='revenue'
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Default channel grouping CASE expression
# (mirrors GA4's default channel group definitions)
# ---------------------------------------------------------------------------

DEFAULT_CHANNEL_GROUPING = """
CASE
    WHEN LOWER(traffic_source.medium) = 'organic'
        THEN 'Organic Search'
    WHEN LOWER(traffic_source.medium) IN ('cpc', 'ppc', 'paidsearch', 'paid search')
        OR (traffic_source.source IN ('google','bing','yahoo','baidu','duckduckgo')
            AND LOWER(traffic_source.medium) = 'cpc')
        THEN 'Paid Search'
    WHEN LOWER(traffic_source.medium) LIKE '%email%'
        OR LOWER(traffic_source.medium) LIKE '%e-mail%'
        OR LOWER(traffic_source.medium) = 'newsletter'
        THEN 'Email'
    WHEN LOWER(traffic_source.medium) IN ('display','cpm','banner','interstitial')
        THEN 'Display'
    WHEN LOWER(traffic_source.medium) IN (
            'social','social-network','social network',
            'social media','sm','social-media','socialmedia')
        OR LOWER(traffic_source.source) IN (
            'facebook','instagram','twitter','linkedin','tiktok',
            'pinterest','snapchat','youtube','reddit','tumblr')
        THEN 'Paid Social'
    WHEN LOWER(traffic_source.medium) = 'affiliate'
        THEN 'Affiliates'
    WHEN LOWER(traffic_source.medium) = 'referral'
        THEN 'Referral'
    WHEN LOWER(traffic_source.medium) IN ('video','paid video')
        THEN 'Video'
    WHEN (traffic_source.source IS NULL
            OR LOWER(traffic_source.source) = '(direct)'
            OR traffic_source.source = '')
        AND (traffic_source.medium IS NULL
            OR LOWER(traffic_source.medium) IN ('(none)', '(not set)', ''))
        THEN 'Direct'
    ELSE CONCAT(
        COALESCE(traffic_source.source, '(direct)'),
        ' / ',
        COALESCE(traffic_source.medium, '(none)')
    )
END
""".strip()

SOURCE_MEDIUM_GROUPING = """
CONCAT(
    COALESCE(traffic_source.source, '(direct)'),
    ' / ',
    COALESCE(traffic_source.medium, '(none)')
)
""".strip()


def _channel_expr(channel_grouping: str) -> str:
    if channel_grouping == "source_medium":
        return SOURCE_MEDIUM_GROUPING
    return DEFAULT_CHANNEL_GROUPING


def _quote_events(events: list[str]) -> str:
    return ", ".join(f"'{e}'" for e in events)


def build_journey_sql(
    project_id: str,
    dataset_id: str,
    start_date: str,
    end_date: str,
    conversion_events: list[str],
    lookback_days: int = 30,
    channel_grouping: str = "default",
) -> str:
    """Build the main customer-journey extraction SQL for GA4.

    Returns rows with one row per (user, conversion, touchpoint).

    Columns returned:
        user_pseudo_id, conversion_timestamp, conversion_value,
        touchpoint_position, total_touchpoints, channel,
        source, medium, campaign, session_timestamp
    """
    channel_expr = _channel_expr(channel_grouping)
    events_list = _quote_events(conversion_events)

    # Use a slightly wider date window for the session lookup so that
    # sessions starting just before start_date are still captured.
    session_start_suffix = _date_minus_days(start_date, lookback_days + 1)

    sql = f"""
-- ============================================================
-- GA4 Multi-Touch Attribution: Journey Extraction
-- Project  : {project_id}
-- Dataset  : {dataset_id}
-- Date     : {start_date} → {end_date}
-- Conversion: {", ".join(conversion_events)}
-- Lookback : {lookback_days} days
-- ============================================================

WITH
-- ── Step 1: Session-level touchpoints ──────────────────────
session_data AS (
    SELECT
        user_pseudo_id,
        (SELECT value.int_value
         FROM UNNEST(event_params)
         WHERE key = 'ga_session_id')               AS session_id,
        MIN(event_timestamp)                          AS session_timestamp_us,
        ANY_VALUE(traffic_source.source)              AS source,
        ANY_VALUE(traffic_source.medium)              AS medium,
        ANY_VALUE(traffic_source.name)                AS campaign,
        ANY_VALUE({channel_expr})                     AS channel
    FROM `{project_id}.{dataset_id}.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '{session_start_suffix}' AND '{end_date}'
        AND event_name = 'session_start'
    GROUP BY user_pseudo_id, session_id
),

-- ── Step 2: Conversion events ──────────────────────────────
conversion_data AS (
    SELECT
        user_pseudo_id,
        event_timestamp                               AS conversion_timestamp_us,
        (SELECT value.int_value
         FROM UNNEST(event_params)
         WHERE key = 'ga_session_id')               AS conversion_session_id,
        -- Revenue: try purchase-specific params first, then generic 'value'
        COALESCE(
            (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'revenue'),
            (SELECT value.float_value  FROM UNNEST(event_params) WHERE key = 'revenue'),
            (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'value'),
            (SELECT value.float_value  FROM UNNEST(event_params) WHERE key = 'value'),
            1.0  -- fallback: count conversions
        )                                             AS conversion_value
    FROM `{project_id}.{dataset_id}.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'
        AND event_name IN ({events_list})
),

-- Deduplicate: one conversion per user per session
conversions AS (
    SELECT
        user_pseudo_id,
        conversion_session_id,
        MIN(conversion_timestamp_us) AS conversion_timestamp_us,
        MAX(conversion_value)        AS conversion_value
    FROM conversion_data
    GROUP BY user_pseudo_id, conversion_session_id
),

-- ── Step 3: Join sessions → conversions within lookback ────
journeys_raw AS (
    SELECT
        c.user_pseudo_id,
        TIMESTAMP_MICROS(c.conversion_timestamp_us)  AS conversion_timestamp,
        c.conversion_value,
        s.session_id,
        TIMESTAMP_MICROS(s.session_timestamp_us)     AS session_timestamp,
        s.source,
        s.medium,
        s.campaign,
        s.channel
    FROM conversions c
    INNER JOIN session_data s
        ON  c.user_pseudo_id         = s.user_pseudo_id
        AND s.session_timestamp_us  <= c.conversion_timestamp_us
        AND s.session_timestamp_us  >= c.conversion_timestamp_us
                                       - {lookback_days} * 86400 * 1000000
)

-- ── Step 4: Final output with position metadata ────────────
SELECT
    user_pseudo_id,
    conversion_timestamp,
    conversion_value,
    session_timestamp,
    source,
    medium,
    campaign,
    channel,
    ROW_NUMBER() OVER (
        PARTITION BY user_pseudo_id, conversion_timestamp
        ORDER BY session_timestamp
    )                                                AS touchpoint_position,
    COUNT(*) OVER (
        PARTITION BY user_pseudo_id, conversion_timestamp
    )                                                AS total_touchpoints
FROM journeys_raw
ORDER BY user_pseudo_id, conversion_timestamp, touchpoint_position
""".strip()

    return sql


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _date_minus_days(date_str: str, days: int) -> str:
    """Subtract days from a YYYYMMDD string and return YYYYMMDD."""
    from datetime import datetime, timedelta  # noqa: PLC0415

    d = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)
    return d.strftime("%Y%m%d")
