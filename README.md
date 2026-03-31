# GA4 Attribution Agent

A conversational AI agent powered by Claude that connects to Google BigQuery, extracts GA4 customer journey data, and runs **7 attribution models** side-by-side so you can compare how credit is assigned across your marketing channels.

---

## Features

- **Conversational setup** — Claude guides you through connecting to your GA4 BigQuery dataset, choosing conversion events, date ranges, and lookback windows
- **Standardized SQL** — Generates a clean, reusable journey-extraction query against GA4's `events_*` tables
- **7 attribution models** run in parallel:

| Model | Description |
|---|---|
| Last Touch | 100% credit to the final touchpoint before conversion |
| First Touch | 100% credit to the first touchpoint |
| Linear | Equal credit split across all touchpoints |
| Time Decay | Exponential decay — more credit to touchpoints closer to conversion |
| Position-Based | 40% first / 40% last / 20% distributed across middle touches (U-shape) |
| Shapley | Game-theory marginal contribution per channel |
| Markov Chain | Transition matrix removal effects (data-driven) |

- **Default channel grouping** — Maps raw `source / medium` into GA4-style groups (Organic Search, Paid Search, Email, Display, Paid Social, Direct, Referral, Affiliates, Video)
- **Source/medium mode** — Optionally keep raw `google / cpc` strings instead

---

## Architecture

```
ga4-attribution-agent/
├── main.py                        # CLI entry point
├── ga4_attribution/
│   ├── agent.py                   # Claude-powered conversational agent (tool use loop)
│   ├── bigquery.py                # BigQuery client wrapper
│   ├── sql_builder.py             # Standardized GA4 journey extraction SQL
│   ├── attribution.py             # All 7 attribution model implementations
│   └── formatters.py              # Table display helpers
└── requirements.txt
```

### How the agent works

```
User ──► Claude (claude-opus-4-6)
              │
              ├─► list_events     (BQ: what events exist?)
              ├─► list_channels   (BQ: what source/medium combos exist?)
              ├─► preview_journeys (BQ: show example multi-touch paths)
              ├─► show_sql        (print the generated SQL)
              └─► run_attribution (BQ extract + all 7 models → results table)
```

Claude asks the right questions, calls BigQuery tools, and explains the results — you just answer in plain English.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

For BigQuery, either:
- Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` in `.env`, or
- Use Application Default Credentials: `gcloud auth application-default login`

### 3. Run the agent

```bash
python main.py
```

The agent will start a conversation and walk you through the setup.

### Preview the SQL without running anything

```bash
python main.py --sql-only
# Or pre-fill your project:
python main.py --sql-only --project my-gcp-project --dataset analytics_123456789
```

---

## Example session

```
Agent: Hello! What is your Google Cloud project ID and BigQuery dataset ID?

You: project is my-company-data, dataset is analytics_987654321

Agent: [calls list_events for 20240101–20240131]
       I can see purchase (4,821 events), add_to_cart (18,304), session_start (142,000)...
       Which event should count as a conversion?

You: purchase

Agent: [calls list_channels]
       Top channels: google/cpc (42k sessions), (direct)/(none) (31k), google/organic (28k)...
       Lookback window? (default 30 days)

You: 30 days is fine

Agent: [calls run_attribution — extracts 4,821 journeys, 19,847 touchpoints]

╒════════════════╤════════════╤═════════════╤═════════╤════════════╤════════════════╤══════════╤════════╕
│ channel        │ last_touch │ first_touch │ linear  │ time_decay │ position_based │ shapley  │ markov │
╞════════════════╪════════════╪═════════════╪═════════╪════════════╪════════════════╪══════════╪════════╡
│ Paid Search    │  48,210    │  12,053     │  28,140 │   41,380   │    33,747      │  31,204  │ 30,891 │
│ Organic Search │  19,284    │  33,747     │  26,327 │   20,161   │    26,515      │  25,840  │ 26,102 │
│ Direct         │  62,673    │   9,642     │  35,621 │   55,441   │    38,568      │  39,042  │ 38,750 │
│ Email          │   8,037    │  16,074     │  12,055 │    9,644   │    12,864      │  11,892  │ 11,940 │
│ Paid Social    │   3,218    │   8,037     │   6,428 │    4,023   │     7,236      │   6,612  │  6,497 │
╘════════════════╧════════════╧═════════════╧═════════╧════════════╧════════════════╧══════════╧════════╛

Agent: Key observations:
- Direct is over-credited by Last Touch (36% vs 22% in Shapley) — likely many users
  returning directly after discovering you through paid channels first.
- Paid Search gets much more credit under Last Touch than First Touch, suggesting it
  closes journeys that Organic Search started.
- Shapley and Markov agree closely, which gives confidence in the data-driven numbers.
```

---

## GA4 BigQuery schema notes

The agent works with the standard GA4 BigQuery export schema:

- **Tables:** `events_YYYYMMDD` (daily partitions, filtered with `_TABLE_SUFFIX`)
- **User key:** `user_pseudo_id`
- **Channel fields:** `traffic_source.source`, `.medium`, `.name` (on `session_start` events)
- **Revenue:** `event_params` key `revenue` or `value` (double/float)
- **Session ID:** `event_params` key `ga_session_id` (int)

---

## Attribution model details

### Shapley Value
Computes each channel's **marginal contribution** by iterating over all possible subsets of channels in a journey. Uses a path-proportional characteristic function `v(S) = value × |S| / |path|`. Falls back to Monte Carlo sampling (200 random permutations) when a journey has more than 15 unique channels.

### Markov Chain
Builds a **transition probability matrix** between channel states, including absorbing states `Conversion` and `Null`. The **removal effect** for each channel = how much the overall conversion probability drops when that channel is removed from the graph. Removal effects are normalised to total conversion value. Falls back to Linear when all journeys in the sample convert (insufficient contrast to estimate counterfactuals).

### Time Decay
Uses exponential decay: `weight = 2^(−Δt / half_life)` where `Δt` is days from touchpoint to conversion. Default half-life is **7 days** (configurable).

### Position-Based
Standard U-shape: **40% first / 40% last / 20% middle**. For 2-touchpoint paths the 20% middle is redistributed proportionally between first and last (50/50 split by default). For single-touch paths, 100% credit.

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A Google Cloud project with GA4 BigQuery export enabled
- BigQuery read permissions on the GA4 dataset

```
anthropic>=0.40.0
google-cloud-bigquery>=3.0.0
google-auth>=2.0.0
db-dtypes>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
tabulate>=0.9.0
python-dotenv>=1.0.0
```
