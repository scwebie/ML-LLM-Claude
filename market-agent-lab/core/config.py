"""Central configuration for market-agent-lab.

All configuration is read from environment variables (optionally loaded
from a local ``.env`` file). Nothing in this module ever reaches out to a
real brokerage, prediction market, or betting service -- see
``docs/architecture.md`` for the enumerated safety boundaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable process-wide settings, loaded once at import time."""

    data_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_STORE_DIR", str(REPO_ROOT / "data_store")))
    )
    duckdb_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DUCKDB_PATH", str(REPO_ROOT / "data_store" / "duckdb" / "market_agent_lab.duckdb"))
        )
    )

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    synthetic_seed: int = field(default_factory=lambda: int(os.getenv("SYNTHETIC_SEED", "42")))
    synthetic_start_date: str = field(
        default_factory=lambda: os.getenv("SYNTHETIC_START_DATE", "2015-01-01")
    )
    synthetic_end_date: str = field(
        default_factory=lambda: os.getenv("SYNTHETIC_END_DATE", "2023-12-31")
    )

    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Hard safety invariant. This is intentionally NOT overridable via
    # environment variable -- v0.1 is paper-trading only, full stop.
    paper_trading_only: bool = True

    def ensure_dirs(self) -> None:
        for sub in ("raw", "features", "models", "duckdb"):
            (self.data_store_dir / sub).mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
