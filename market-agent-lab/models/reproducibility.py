"""V0.3 Stage 13: model registry reproducibility.

Every trained model's registry record should carry enough provenance to
answer, without re-running anything, "was this number produced by the
code and data I think it was?": the git commit that produced it, a hash
of the target-definition source code (so a silent change to how targets
are computed is detectable even though ``model_version`` alone would not
show it), the random seed actually used, a fingerprint of the exact
training+validation data fed into LightGBM, and a hash of the saved
model artifacts themselves. None of this changes what gets trained --
it only records enough to verify reproducibility after the fact.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit(repo_dir: Path | None = None) -> str | None:
    """Best-effort only -- returns ``None`` (never raises) if this isn't a
    git checkout or git isn't available, e.g. a packaged deploy."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir or _REPO_ROOT),
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def hash_source(func: Callable) -> str:
    """SHA-256 of a function's own source text -- fingerprints target-
    definition logic so a silent behavioural change is detectable even
    when ``model_version`` alone would look identical."""
    return hashlib.sha256(inspect.getsource(func).encode("utf-8")).hexdigest()


def compute_data_fingerprint(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]
) -> str:
    """Deterministic SHA-256 fingerprint of the exact rows/columns fed into
    training and validation -- catches a silent change in the data (a
    different row count, a shifted date range, a changed feature value)
    even when every other registry field stays the same."""
    columns = [*feature_cols, *target_cols]
    digest = hashlib.sha256()
    for label, df in (("train", train_df), ("val", val_df)):
        present = [c for c in columns if c in df.columns]
        ordered = df.loc[:, present]
        digest.update(f"{label}|shape={ordered.shape[0]}x{ordered.shape[1]}|columns={','.join(present)}|".encode())
        if not ordered.empty:
            digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
    return digest.hexdigest()


def compute_artifact_hash(artifact_dir: Path) -> str:
    """SHA-256 over every booster ``.txt`` file's bytes in a fixed (sorted)
    order -- verifies the persisted model artifact itself, not just the
    metadata describing how it was supposedly produced."""
    digest = hashlib.sha256()
    for path in sorted(Path(artifact_dir).glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def extract_seed(hyperparameters: dict) -> int | None:
    """Pulls the LightGBM ``seed`` out of the (per-target-kind)
    hyperparameter dict for direct, queryable storage -- it is already
    inside ``hyperparameters_json`` but buried a level deep."""
    for params in hyperparameters.values():
        if isinstance(params, dict) and "seed" in params:
            return int(params["seed"])
    return None
