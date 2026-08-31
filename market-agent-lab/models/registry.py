"""Model registry: versioning, artifact storage, and champion/challenger role tracking.

Every trained model is saved to ``data_store/models/<model_version>/`` as
one LightGBM text file per target plus a ``metadata.json``, and a matching
row is written to the ``model_registry`` DuckDB table (see
``database/schema.py``) recording hyperparameters, training/validation/test
periods, feature version, feature names, and metrics. Nothing about a
registered model is ever mutated in place -- promotions/demotions only
change the ``role`` column; retraining always creates a brand-new
``model_version``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import lightgbm as lgb

from core.config import settings
from database import repository as repo
from models.train import TrainedModels

ROLE_CHAMPION = "CHAMPION"
ROLE_CHALLENGER = "CHALLENGER"
ROLE_ARCHIVED = "ARCHIVED"


@dataclass
class ModelPeriods:
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime | None = None
    test_end: datetime | None = None


def _next_model_version(con: duckdb.DuckDBPyConnection) -> str:
    row = con.execute("SELECT COUNT(*) FROM model_registry").fetchone()
    existing = row[0] if row is not None else 0
    return f"lgbm_v{existing + 1:04d}"


def register_model(
    con: duckdb.DuckDBPyConnection,
    trained: TrainedModels,
    feature_version: str,
    periods: ModelPeriods,
    metrics: dict,
    role: str = ROLE_CHALLENGER,
    model_version: str | None = None,
) -> str:
    model_version = model_version or _next_model_version(con)
    artifact_dir = settings.data_store_dir / "models" / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for target_name, booster in trained.boosters.items():
        booster.save_model(str(artifact_dir / f"{target_name}.txt"))

    metadata = {
        "model_version": model_version,
        "feature_names": trained.feature_names,
        "hyperparameters": trained.hyperparameters,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    record = {
        "model_version": model_version,
        "role": role,
        "created_at": datetime.now(UTC),
        "feature_version": feature_version,
        "feature_names": trained.feature_names,
        "hyperparameters": trained.hyperparameters,
        "training_period_start": periods.training_start,
        "training_period_end": periods.training_end,
        "validation_period_start": periods.validation_start,
        "validation_period_end": periods.validation_end,
        "test_period_start": periods.test_start,
        "test_period_end": periods.test_end,
        "metrics": metrics,
        "artifact_path": str(artifact_dir),
    }
    repo.upsert_model_registry(con, record)
    return model_version


def load_model(con: duckdb.DuckDBPyConnection, model_version: str) -> tuple[dict[str, lgb.Booster], dict]:
    df = repo.get_model_registry(con)
    row = df[df["model_version"] == model_version]
    if row.empty:
        raise KeyError(f"model_version={model_version} not found in registry")
    record = row.iloc[0].to_dict()
    artifact_dir = Path(record["artifact_path"])
    feature_names = json.loads(record["feature_names_json"])

    boosters = {}
    for target_name in ("excess_return_5d", "excess_return_20d", "positive_5d", "positive_20d"):
        model_path = artifact_dir / f"{target_name}.txt"
        if model_path.exists():
            boosters[target_name] = lgb.Booster(model_file=str(model_path))
    record["feature_names"] = feature_names
    record["metrics"] = json.loads(record["metrics_json"])
    return boosters, record


def get_champion(con: duckdb.DuckDBPyConnection) -> dict | None:
    return repo.get_champion(con)


def set_role(con: duckdb.DuckDBPyConnection, model_version: str, role: str) -> None:
    con.execute("UPDATE model_registry SET role = ? WHERE model_version = ?", [role, model_version])


def promote_to_champion(con: duckdb.DuckDBPyConnection, model_version: str) -> None:
    """Demote the current champion (if any) to ARCHIVED and promote ``model_version``."""
    current = get_champion(con)
    if current is not None and current["model_version"] != model_version:
        set_role(con, current["model_version"], ROLE_ARCHIVED)
    set_role(con, model_version, ROLE_CHAMPION)
