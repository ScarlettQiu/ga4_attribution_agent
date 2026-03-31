"""BigQuery client wrapper for GA4 data access."""

from __future__ import annotations
import json
from typing import Any
import pandas as pd


class BigQueryClient:
    """Thin wrapper around google.cloud.bigquery that returns JSON-serializable results."""

    def __init__(self, project_id: str | None = None, credentials=None):
        try:
            from google.cloud import bigquery  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "google-cloud-bigquery is required. "
                "Run: pip install google-cloud-bigquery db-dtypes"
            ) from e

        self.project_id = project_id
        self._client = bigquery.Client(project=project_id, credentials=credentials)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run_query(self, sql: str, max_rows: int = 50_000) -> pd.DataFrame:
        """Execute SQL and return a DataFrame. Raises on query error."""
        job = self._client.query(sql)
        df = job.result().to_dataframe()
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df

    def run_query_to_json(self, sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
        """Execute SQL and return JSON-serializable rows (for Claude tools)."""
        df = self.run_query(sql, max_rows=max_rows)
        # Convert Timestamp / numpy types so json.dumps won't choke
        return json.loads(df.to_json(orient="records", date_format="iso"))

    # ------------------------------------------------------------------
    # GA4-specific helpers
    # ------------------------------------------------------------------

    def list_events(
        self,
        project_id: str,
        dataset_id: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Return event names with counts and example params."""
        sql = f"""
        SELECT
            event_name,
            COUNT(*) AS event_count,
            COUNT(DISTINCT user_pseudo_id) AS unique_users
        FROM `{project_id}.{dataset_id}.events_*`
        WHERE _TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY event_name
        ORDER BY event_count DESC
        LIMIT 50
        """
        return self.run_query_to_json(sql)

    def list_channels(
        self,
        project_id: str,
        dataset_id: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Return source/medium combinations with session counts."""
        sql = f"""
        SELECT
            traffic_source.source AS source,
            traffic_source.medium AS medium,
            COUNT(DISTINCT user_pseudo_id) AS users,
            COUNT(*) AS sessions
        FROM `{project_id}.{dataset_id}.events_*`
        WHERE _TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'
            AND event_name = 'session_start'
        GROUP BY source, medium
        ORDER BY sessions DESC
        LIMIT 50
        """
        return self.run_query_to_json(sql)

    def preview_journeys(
        self,
        project_id: str,
        dataset_id: str,
        start_date: str,
        end_date: str,
        conversion_events: list[str],
        lookback_days: int = 30,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return a few example multi-touch journeys for inspection."""
        from .sql_builder import build_journey_sql  # noqa: PLC0415

        sql = build_journey_sql(
            project_id=project_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            conversion_events=conversion_events,
            lookback_days=lookback_days,
        )
        full_sql = f"""
        WITH journeys AS ({sql})
        SELECT *
        FROM journeys
        WHERE user_pseudo_id IN (
            SELECT DISTINCT user_pseudo_id
            FROM journeys
            LIMIT {limit}
        )
        ORDER BY user_pseudo_id, conversion_timestamp, touchpoint_position
        """
        return self.run_query_to_json(full_sql, max_rows=limit * 20)

    def extract_journeys(
        self,
        project_id: str,
        dataset_id: str,
        start_date: str,
        end_date: str,
        conversion_events: list[str],
        lookback_days: int = 30,
        channel_grouping: str = "default",
    ) -> pd.DataFrame:
        """Run the full journey extraction SQL and return a DataFrame."""
        from .sql_builder import build_journey_sql  # noqa: PLC0415

        sql = build_journey_sql(
            project_id=project_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            conversion_events=conversion_events,
            lookback_days=lookback_days,
            channel_grouping=channel_grouping,
        )
        return self.run_query(sql)
