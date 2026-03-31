"""Attribution model implementations.

Input: a pandas DataFrame with columns:
    user_pseudo_id, conversion_timestamp, conversion_value,
    touchpoint_position, total_touchpoints, channel, session_timestamp

Each row = one touchpoint on the path to one conversion.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "last_touch",
    "first_touch",
    "linear",
    "time_decay",
    "position_based",
    "shapley",
    "markov",
]


def run_all_models(
    journeys: pd.DataFrame,
    models: list[str] | None = None,
    time_decay_half_life_days: float = 7.0,
    position_first: float = 0.40,
    position_last: float = 0.40,
) -> pd.DataFrame:
    """Run attribution models and return a summary DataFrame.

    Returns columns: channel, <model_1>, <model_2>, ...
    Values are the attributed conversion value (float).
    """
    if journeys.empty:
        return pd.DataFrame(columns=["channel"] + (models or AVAILABLE_MODELS))

    models = models or AVAILABLE_MODELS
    results: dict[str, pd.Series] = {}

    fn_map = {
        "last_touch":     lambda j: last_touch(j),
        "first_touch":    lambda j: first_touch(j),
        "linear":         lambda j: linear(j),
        "time_decay":     lambda j: time_decay(j, half_life_days=time_decay_half_life_days),
        "position_based": lambda j: position_based(j, first_weight=position_first, last_weight=position_last),
        "shapley":        lambda j: shapley(j),
        "markov":         lambda j: markov(j),
    }

    for model_name in models:
        if model_name not in fn_map:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {AVAILABLE_MODELS}")
        results[model_name] = fn_map[model_name](journeys)

    combined = pd.DataFrame(results).fillna(0).round(2)
    combined.index.name = "channel"
    combined = combined.reset_index()
    return combined.sort_values(combined.columns[1], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Individual models
# (all take a journeys DataFrame, return a Series indexed by channel)
# ---------------------------------------------------------------------------

def last_touch(journeys: pd.DataFrame) -> pd.Series:
    """100% credit to the last touchpoint before conversion."""
    last = (
        journeys
        .sort_values("touchpoint_position")
        .groupby(["user_pseudo_id", "conversion_timestamp"])
        .last()
        .reset_index()
    )
    return last.groupby("channel")["conversion_value"].sum().rename("last_touch")


def first_touch(journeys: pd.DataFrame) -> pd.Series:
    """100% credit to the first touchpoint."""
    first = (
        journeys
        .sort_values("touchpoint_position")
        .groupby(["user_pseudo_id", "conversion_timestamp"])
        .first()
        .reset_index()
    )
    return first.groupby("channel")["conversion_value"].sum().rename("first_touch")


def linear(journeys: pd.DataFrame) -> pd.Series:
    """Equal credit split among all touchpoints in the path."""
    df = journeys.copy()
    df["credit"] = df["conversion_value"] / df["total_touchpoints"]
    return df.groupby("channel")["credit"].sum().rename("linear")


def time_decay(
    journeys: pd.DataFrame,
    half_life_days: float = 7.0,
) -> pd.Series:
    """Exponential decay: more credit to touchpoints closer to conversion.

    Weight = 2^(Δt / half_life) where Δt = days to conversion (0 = same day).
    """
    df = journeys.copy()

    # Seconds from touchpoint to conversion
    delta_s = (
        df["conversion_timestamp"] - df["session_timestamp"]
    ).dt.total_seconds().clip(lower=0)
    delta_days = delta_s / 86_400.0

    # Recency weight (higher = more recent)
    df["raw_weight"] = np.power(2.0, -delta_days / half_life_days)

    # Normalise per journey
    journey_key = ["user_pseudo_id", "conversion_timestamp"]
    df["weight_sum"] = df.groupby(journey_key)["raw_weight"].transform("sum")
    df["credit"] = df["conversion_value"] * (df["raw_weight"] / df["weight_sum"])

    return df.groupby("channel")["credit"].sum().rename("time_decay")


def position_based(
    journeys: pd.DataFrame,
    first_weight: float = 0.40,
    last_weight: float = 0.40,
) -> pd.Series:
    """U-shaped (position-based) model.

    first_weight → first touchpoint
    last_weight  → last touchpoint
    1 - first_weight - last_weight → split equally among middle touchpoints
    """
    middle_weight = 1.0 - first_weight - last_weight
    df = journeys.copy()

    n = df["total_touchpoints"]
    pos = df["touchpoint_position"]

    # Assign weights
    is_first = pos == 1
    is_last = pos == n
    is_middle = (~is_first) & (~is_last)
    n_middle = (n - 2).clip(lower=0)

    # When n == 2 there is no middle; redistribute middle weight proportionally
    # so the path still sums to 1.0.
    first_w2 = first_weight / (first_weight + last_weight)
    last_w2  = last_weight  / (first_weight + last_weight)

    weight = np.where(
        n == 1,
        1.0,
        np.where(
            n == 2,
            np.where(is_first, first_w2, last_w2),
            # n >= 3
            np.where(
                is_first,
                first_weight,
                np.where(
                    is_last,
                    last_weight,
                    np.where(n_middle > 0, middle_weight / n_middle, 0.0),
                ),
            ),
        ),
    )

    df["credit"] = df["conversion_value"] * weight
    return df.groupby("channel")["credit"].sum().rename("position_based")


def shapley(journeys: pd.DataFrame) -> pd.Series:
    """Shapley value attribution based on marginal channel contributions.

    For each journey, computes the Shapley value for each channel using
    the characteristic function v(S) = conversion_value if S is a subset
    of the actual path, 0 otherwise (path-based Shapley).

    Falls back to a data-driven approach when many unique channels exist.
    """
    # Group journeys by (user, conversion) and get ordered channel list
    journey_groups = (
        journeys
        .sort_values("touchpoint_position")
        .groupby(["user_pseudo_id", "conversion_timestamp"])
        .agg(
            channels=("channel", list),
            conversion_value=("conversion_value", "first"),
        )
        .reset_index()
    )

    unique_channels = journeys["channel"].unique()
    n_channels = len(unique_channels)

    if n_channels > 15:
        # Monte Carlo Shapley approximation
        return _shapley_monte_carlo(journey_groups, n_permutations=200)

    return _shapley_exact(journey_groups)


def _shapley_exact(journey_groups: pd.DataFrame) -> pd.Series:
    """Exact Shapley: enumerate all coalitions for each journey."""
    channel_credits: dict[str, float] = {}

    for _, row in journey_groups.iterrows():
        path = list(dict.fromkeys(row["channels"]))  # deduplicate, preserve order
        n = len(path)
        value = row["conversion_value"]

        # Characteristic function: full path value, partial = proportional
        # v(S) = value * (|S| / |path|) — linear path-based heuristic
        def v(coalition: frozenset) -> float:
            return value * len(coalition) / n

        for channel in path:
            others = [c for c in path if c != channel]
            phi = 0.0
            for size in range(len(others) + 1):
                for subset in combinations(others, size):
                    s = frozenset(subset)
                    s_with = s | {channel}
                    n_s = len(s)
                    weight = _factorial(n_s) * _factorial(n - n_s - 1) / _factorial(n)
                    phi += weight * (v(s_with) - v(s))
            channel_credits[channel] = channel_credits.get(channel, 0.0) + phi

    return pd.Series(channel_credits, name="shapley")


def _shapley_monte_carlo(
    journey_groups: pd.DataFrame,
    n_permutations: int = 200,
) -> pd.Series:
    """Monte Carlo Shapley approximation via random permutations."""
    rng = np.random.default_rng(42)
    channel_credits: dict[str, float] = {}

    for _, row in journey_groups.iterrows():
        path = list(dict.fromkeys(row["channels"]))
        n = len(path)
        value = row["conversion_value"]

        marginals: dict[str, list[float]] = {c: [] for c in path}

        for _ in range(n_permutations):
            perm = rng.permutation(path).tolist()
            for i, channel in enumerate(perm):
                # Marginal contribution = v(first i+1) - v(first i)
                before = value * i / n
                after = value * (i + 1) / n
                marginals[channel].append(after - before)

        for channel in path:
            avg = float(np.mean(marginals[channel])) if marginals[channel] else 0.0
            channel_credits[channel] = channel_credits.get(channel, 0.0) + avg

    return pd.Series(channel_credits, name="shapley")


def markov(journeys: pd.DataFrame) -> pd.Series:
    """Markov Chain attribution (often called data-driven or Bayesian).

    Builds a transition matrix from touchpoint states (channels) including
    a 'Start', 'Conversion', and 'Null' (no conversion) state.
    Attribution = removal effect: how much conversion rate drops when
    a channel is removed from the graph.
    """
    START = "__start__"
    CONV = "__conversion__"
    NULL = "__null__"

    # Build transition counts
    transitions: dict[tuple[str, str], int] = {}

    journey_groups = (
        journeys
        .sort_values(["user_pseudo_id", "conversion_timestamp", "touchpoint_position"])
        .groupby(["user_pseudo_id", "conversion_timestamp"])
        .agg(
            channels=("channel", list),
            converted=("conversion_value", lambda x: x.sum() > 0),
        )
        .reset_index()
    )

    for _, row in journey_groups.iterrows():
        path = row["channels"]
        converted = row["converted"]
        chain = [START] + path + ([CONV] if converted else [NULL])
        for i in range(len(chain) - 1):
            key = (chain[i], chain[i + 1])
            transitions[key] = transitions.get(key, 0) + 1

    # Ensure NULL state always exists (needed for channel removal step)
    if NULL not in {s for pair in transitions for s in pair}:
        transitions[(NULL, NULL)] = 0

    all_states = sorted(
        {s for pair in transitions for s in pair}
    )

    # Build transition probability matrix
    idx = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    T = np.zeros((n, n))

    for (fr, to), count in transitions.items():
        T[idx[fr], idx[to]] += count

    # Row-normalise (absorbing states stay absorbing)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    conv_idx = idx.get(CONV)
    start_idx = idx.get(START)
    if conv_idx is None or start_idx is None:
        return pd.Series(dtype=float, name="markov")

    # Conversion probability from Start
    total_conv_prob = _absorption_prob(T, start_idx, conv_idx, len(all_states))

    # Removal effect for each channel
    channels = [s for s in all_states if s not in (START, CONV, NULL)]
    removal_effects: dict[str, float] = {}

    for ch in channels:
        ch_idx = idx[ch]
        T_removed = T.copy()
        # Remove channel: redirect all transitions to/from ch to NULL
        T_removed[ch_idx, :] = 0
        T_removed[ch_idx, idx[NULL]] = 1.0
        # Incoming transitions → NULL
        for s in range(n):
            if s != ch_idx and T_removed[s, ch_idx] > 0:
                T_removed[s, idx[NULL]] += T_removed[s, ch_idx]
                T_removed[s, ch_idx] = 0

        conv_prob_removed = _absorption_prob(T_removed, start_idx, conv_idx, n)
        removal_effects[ch] = max(0.0, total_conv_prob - conv_prob_removed)

    # Normalise removal effects → attributed conversion value
    total_removal = sum(removal_effects.values())
    if total_removal == 0:
        # No contrast available (e.g. all journeys convert) — fall back to linear
        return linear(journeys).rename("markov")

    total_conversions = journeys.groupby(
        ["user_pseudo_id", "conversion_timestamp"]
    )["conversion_value"].first().sum()

    channel_credits = {
        ch: (re / total_removal) * total_conversions
        for ch, re in removal_effects.items()
    }
    return pd.Series(channel_credits, name="markov")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _absorption_prob(T: np.ndarray, start: int, absorb: int, n: int) -> float:
    """Probability of reaching absorbing state `absorb` from `start`."""
    # Simple power-iteration approach
    state = np.zeros(n)
    state[start] = 1.0
    for _ in range(500):
        new_state = state @ T
        if abs(new_state[absorb] - state[absorb]) < 1e-8:
            break
        state = new_state
    return float(state[absorb])


def _factorial(n: int) -> float:
    import math  # noqa: PLC0415
    return float(math.factorial(n))
