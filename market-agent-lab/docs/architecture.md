# Architecture (Version 0.1)

market-agent-lab is a **paper-trading / simulation-only** multi-agent
quantitative research system. This document describes the fifteen core
components, how data flows between them, and the hard safety boundaries
that hold at every layer.

## Component map

| # | Component | Module |
|---|-----------|--------|
| 1 | Orchestrator Agent | `agents/orchestrator.py` |
| 2 | Technical Analysis Agent | `agents/technical.py` |
| 3 | Fundamental Analysis Agent | `agents/fundamental.py` |
| 4 | Market Overview Agent | `agents/market_overview.py` |
| 5 | Historical Research Agent | `agents/historical.py` |
| 6 | News / Event Intelligence Agent | `agents/event_intelligence.py` |
| 7 | Feature Store | `features/feature_store.py` |
| 8 | ML Alpha Model | `models/train.py`, `models/predict.py` |
| 9 | Portfolio Decision Engine | `portfolio/allocation.py` |
| 10 | Deterministic Risk Engine | `portfolio/risk.py` |
| 11 | Paper Execution Engine | `execution/paper.py` |
| 12 | Performance Database | `database/` (DuckDB) |
| 13 | Model Training / Retraining Engine | `models/`, `learning/retrain.py` |
| 14 | Backtesting Engine | `backtesting/` |
| 15 | Monitoring Dashboard | `dashboard/app.py`, `api/main.py` |

## Data flow

```mermaid
flowchart TD
    subgraph Data["Raw Data (synthetic, v0.1)"]
        MD[Market OHLCV]
        FD[Fundamentals]
        MAC[Macro series]
        NEWS[News / event sentiment]
    end

    subgraph Features["Deterministic Feature Engines"]
        TF[Technical features]
        FF[Fundamental features]
        MF[Macro features]
        HF[Historical similarity]
    end

    subgraph Agents["Research Agents (Orchestrator-coordinated)"]
        TA[Technical Agent]
        FA[Fundamental Agent]
        MOA[Market Overview Agent]
        HRA[Historical Research Agent]
        EIA[Event Intelligence Agent]
    end

    FS[(Feature Store\nDuckDB: feature_snapshots)]

    MODEL[ML Alpha Model\nLightGBM x4 targets]
    PRED[(Model Predictions\nimmutable)]

    PDE[Portfolio Decision Engine\ndeterministic]
    RISK{Deterministic Risk Engine\nAPPROVED / REJECTED}
    EXEC[Paper Execution Engine\nsimulated broker]
    PERF[(Performance Database\nfills, orders, snapshots)]

    OUT[Outcome Labelling]
    RETRAIN[Retraining + Champion/Challenger]

    MD --> TF
    FD --> FF
    MAC --> MF
    MD --> HF

    TF --> TA
    FF --> FA
    MF --> MOA
    HF --> HRA
    NEWS --> EIA

    TA --> FS
    FA --> FS
    MOA --> FS
    HRA --> FS
    EIA --> FS
    TF --> FS
    FF --> FS
    MF --> FS

    FS --> MODEL
    MODEL --> PRED
    PRED --> PDE
    PDE -->|proposed orders| RISK
    RISK -->|APPROVED| EXEC
    RISK -->|REJECTED + reason code| PERF
    EXEC --> PERF
    PERF --> OUT
    OUT --> RETRAIN
    RETRAIN -->|challenger| MODEL
    RETRAIN -->|promote/reject decision, logged| MODEL

    DASH[Monitoring Dashboard]
    PERF --> DASH
    PRED --> DASH
    FS --> DASH
```

## Hard safety boundaries

These are architectural invariants, enforced by code structure, not just
convention:

* **Simulation only.** `execution/paper.py` is a fully in-process simulated
  broker. There is no HTTP client, SDK, or credential anywhere in the
  codebase that talks to a real brokerage, exchange, or bank.
* **No prediction markets / betting.** `agents/event_intelligence.py`
  processes ordinary financial/economic news and scheduled corporate
  events only; no such data source is integrated anywhere.
* **Agents never place orders.** Every agent module (`agents/*.py`)
  returns a Pydantic `AgentReport` (or an agent-specific structured
  output) -- none of them import `execution/` or `portfolio/risk.py`.
  Order construction lives entirely in `portfolio/allocation.py`.
* **The Risk Engine is the only gate to execution.** `execution/orders.py`
  routes every proposed order through
  `portfolio.risk.RiskEngine.evaluate_order` before
  `execution/paper.py` will fill anything. There is no code path from a
  model prediction or an agent output directly into a fill.
* **The Risk Engine is deterministic.** `portfolio/risk.py` contains only
  arithmetic and comparisons against configured `RiskLimits` -- no LLM
  call, no learned model, no randomness.
* **Kill switch is explicit and sticky.** Once
  `RiskEngine.engage_kill_switch()` fires (from a daily-loss or drawdown
  breach), every subsequent order is rejected with `KILL_SWITCH` until a
  human operator calls `reset_kill_switch()` -- no agent, model, or
  automated process can reset it.
* **Predictions and fills are immutable.** `ModelPrediction`, `Outcome`,
  and `PaperFill` (`core/schemas.py`) are frozen Pydantic models;
  `database/repository.py` raises if you try to insert a prediction or
  outcome whose ID already exists.
* **No uncontrolled self-modification.** `learning/retrain.py` only ever
  produces a new `CHALLENGER` model version. Promotion to `CHAMPION`
  happens exclusively via `learning/champion_challenger.py`'s
  multi-criteria gate, and every decision (promote or reject, with
  rationale) is written to `promotion_log`.
* **Everything is recorded.** Agent reports, feature snapshots, model
  versions, predictions, proposed orders (approved and rejected, with
  reason codes), fills, and portfolio snapshots are all persisted to
  DuckDB (see `database/schema.py`).

## Why DuckDB, and how to migrate to PostgreSQL/TimescaleDB later

`database/db.py` isolates all connection handling behind
`get_connection()`, and `database/schema.py` uses only portable SQL (no
DuckDB-specific types) for every application table. Migrating to
PostgreSQL/TimescaleDB in a later version means rewriting `db.py`'s
connection object and `schema.py`'s DDL dialect; `database/repository.py`
and every caller above it should not need to change.

## Design deviation: no `openai-agents` dependency

The OpenAI Agents SDK's PyPI package (`openai-agents`) imports as `import
agents`, which collides with this project's required top-level `agents/`
package. `agents/base.py` implements a small in-house orchestration layer
that mirrors the Agents SDK's structured-input/structured-output pattern
using the plain `openai` client directly, gated so an LLM call only ever
contributes an optional natural-language `reasoning_summary` -- every
numeric score is deterministic. See `agents/base.py` for the full
rationale.
