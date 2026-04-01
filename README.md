# GA4 Multi-Touch Attribution Skill

A complete GA4 marketing attribution toolkit powered by Claude and Google BigQuery. Compares **7 attribution models** side-by-side across four interfaces — from a conversational CLI agent to an interactive Streamlit web app to a Claude Code skill and a LangChain SQL agent.

---

## Architecture

```mermaid
graph TB
    subgraph Interfaces["Entry Points"]
        A["🖥️  CLI Agent\nmain.py"]
        B["🌐  Streamlit UI\napp.py"]
        C["⌨️  Claude Code Skill\n/ga4-attribution"]
        D["🔗  LangChain SQL Agent\nlangchain_agent.py"]
    end

    subgraph Core["Core Library  ga4_attribution/"]
        E["agent.py\nClaude tool-use loop"]
        F["bigquery.py\nBigQueryClient"]
        G["sql_builder.py\nbuild_journey_sql()"]
        H["attribution.py\n7 models"]
        I["streamlit_tools.py\nexecute_tool()"]
    end

    subgraph BQ["Google Cloud"]
        J[("BigQuery\nevents_* tables")]
    end

    subgraph LLM["Anthropic"]
        K["Claude\nclaude-opus-4-6"]
    end

    A --> E
    B --> I
    C --> F
    C --> H
    D --> F
    D --> H

    E --> K
    E --> F
    B --> K
    D --> K

    I --> F
    F --> G
    F --> J
    G --> J
    H --> F
```

### How the four interfaces work

| Interface | How to run | Best for |
|---|---|---|
| **CLI Agent** (`main.py`) | `python main.py` | Guided conversational analysis |
| **Streamlit UI** (`app.py`) | `streamlit run app.py` | Visual exploration, charts, sharing |
| **Claude Code Skill** | `/ga4-attribution` in `claude` CLI | Quick runs from your terminal |
| **LangChain SQL Agent** | `python langchain_agent.py` | Ad-hoc SQL + attribution in one chat |

---

## Features

- **Conversational setup** — Claude guides you through dataset, conversion events, date ranges, and lookback windows
- **Standardized SQL** — Generates a clean 4-CTE journey-extraction query against GA4's `events_*` tables
- **7 attribution models** run in parallel:

| Model | Description |
|---|---|
| Last Touch | 100% credit to the final touchpoint before conversion |
| First Touch | 100% credit to the first touchpoint |
| Linear | Equal credit split across all touchpoints |
| Time Decay | Exponential decay — more credit to touchpoints closer to conversion |
| Position-Based | 40% first / 40% last / 20% middle (U-shape) |
| Shapley | Game-theory marginal contribution per channel |
| Markov Chain | Transition matrix removal effects (data-driven) |

- **Default channel grouping** — Maps raw `source / medium` into GA4-style groups (Organic Search, Paid Search, Email, Display, Paid Social, Direct, Referral, Affiliates, Video)
- **Source/medium mode** — Optionally keep raw `google / cpc` strings

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/ScarlettQiu/ga4_attribution_skill.git
cd ga4_attribution_skill
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

For BigQuery:
```bash
gcloud auth application-default login
# OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json in .env
```

### 3. Pick your interface

**CLI Agent** (guided conversation):
```bash
python main.py
```

**Streamlit UI** (visual, charts):
```bash
streamlit run app.py
```

**LangChain SQL Agent** (conversational SQL + attribution):
```bash
python langchain_agent.py --project my-project --dataset analytics_123456789
```

**Claude Code Skill** (from terminal `claude` session):
```
/ga4-attribution
```

### Preview SQL without running anything

```bash
python run_attribution.py \
  --project my-project \
  --dataset analytics_123 \
  --start 20240101 --end 20240131 \
  --events purchase \
  --sql-only
```

---

## Streamlit UI

The web app (`app.py`) adds:
- **Streaming responses** — see Claude's thinking in real time
- **Interactive Plotly charts** — grouped bar chart comparing all 7 models per channel
- **Artifact persistence** — charts and tables stay visible as the conversation continues
- **Streamlit Cloud secrets** — configure API key and GCP service account via `st.secrets`

To deploy on Streamlit Cloud, add these secrets in the dashboard:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."

[gcp_service_account]
type = "service_account"
project_id = "my-project"
private_key_id = "..."
private_key = "-----BEGIN RSA PRIVATE KEY-----\n..."
client_email = "..."
```

---

## LangChain SQL Agent

`langchain_agent.py` wraps BigQuery in LangChain's `SQLDatabaseToolkit` so Claude can write and iterate on GA4 SQL freely alongside our attribution models.

**SQL tools available:**
- `sql_db_list_tables` — list available event tables
- `sql_db_schema` — get column schema
- `sql_db_query` — run arbitrary SQL
- `sql_db_query_checker` — validate SQL before running

**Custom tool:**
- `run_attribution_models` — runs all 7 models end-to-end

Example questions the LangChain agent can answer:
```
"What conversion events are available last month?"
"Show me the top 10 source/medium combinations by sessions"
"Run attribution analysis for purchase events in January 2024"
"How many multi-touch journeys had 3+ touchpoints?"
```

---

## Claude Code Skill

Install once — use anywhere in your `claude` CLI sessions:

```bash
# The skill file is already at:
~/.claude/commands/ga4-attribution.md
```

Then in any `claude` terminal session:
```
/ga4-attribution
```

Claude will ask for your project, dataset, and date range, run the analysis, and explain the results.

> **Note:** Skills work in the `claude` terminal CLI only, not in IDE extensions.

---

## Testing with Public Data

Google provides an obfuscated GA4 sample dataset in BigQuery:

| Parameter | Value |
|---|---|
| Project | your own GCP project ID (for billing) |
| Dataset | `bigquery-public-data.ga4_obfuscated_sample_ecommerce` |
| Dates | `20201101` → `20210131` |
| Event | `purchase` |

No setup required beyond `gcloud auth application-default login`.

---

## GA4 BigQuery Schema Notes

- **Tables:** `events_YYYYMMDD` — always filter with `_TABLE_SUFFIX BETWEEN 'start' AND 'end'`
- **User key:** `user_pseudo_id`
- **Channel fields:** `traffic_source.source`, `.medium`, `.name` on `session_start` events
- **Event params:** accessed via `UNNEST(event_params)` — never direct array indexing
- **Revenue:** `event_params` key `revenue` or `value` (double/float, with COALESCE fallback)
- **Session ID:** `event_params` key `ga_session_id` (int)

---

## Attribution Model Details

### Shapley Value
Computes each channel's **marginal contribution** by iterating over all possible subsets of channels in a journey. Uses a path-proportional characteristic function `v(S) = value × |S| / |path|`. Falls back to Monte Carlo sampling (200 random permutations) when a journey has more than 15 unique channels.

### Markov Chain
Builds a **transition probability matrix** between channel states, including absorbing states `Conversion` and `Null`. The **removal effect** for each channel = how much the overall conversion probability drops when that channel is removed. Removal effects are normalised to total conversion value. Falls back to Linear when all journeys convert (no contrast for counterfactuals).

### Time Decay
Uses exponential decay: `weight = 2^(−Δt / half_life)` where `Δt` is days from touchpoint to conversion. Default half-life is **7 days**.

### Position-Based
Standard U-shape: **40% first / 40% last / 20% middle**. For 2-touchpoint paths the middle weight redistributes proportionally (50/50). Single-touch paths get 100%.

---

## Project Structure

```
ga4_attribution_skill/
├─��� main.py                    # CLI agent entry point
├── app.py                     # Streamlit web UI
├── run_attribution.py         # Standalone CLI (used by the skill)
├── langchain_agent.py         # LangChain SQL agent
├── ga4_attribution/
│   ├── __init__.py
│   ├── agent.py               # Claude tool-use loop + tools
│   ├── bigquery.py            # BigQueryClient wrapper
│   ├── sql_builder.py         # 4-CTE journey extraction SQL
│   ├── attribution.py         # All 7 attribution models
│   └── streamlit_tools.py     # execute_tool() for Streamlit artifacts
├── .streamlit/
│   ├── config.toml            # Theme settings
│   └── secrets.toml.example   # Secrets template
├── requirements.txt
└── .env.example
```

---

## Requirements

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/)
- Google Cloud project with GA4 BigQuery export enabled
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
streamlit>=1.35.0
plotly>=5.20.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-community>=0.3.0
sqlalchemy-bigquery>=1.9.0
```
