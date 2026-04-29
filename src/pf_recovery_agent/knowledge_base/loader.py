"""Knowledge base loader – reads JSON data files into typed model instances."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pf_recovery_agent.models import RCA, TSG, ServiceDependency


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_service_dependencies(data_dir: Path | None = None) -> list[ServiceDependency]:
    """Load all service dependency records from the data directory."""
    from pf_recovery_agent.config import DATA_DIR

    path = (data_dir or DATA_DIR) / "service_dependencies.json"
    return [ServiceDependency(**item) for item in _load_json(path)]


@lru_cache(maxsize=1)
def load_rcas(data_dir: Path | None = None) -> list[RCA]:
    """Load all RCA records from the data directory."""
    from pf_recovery_agent.config import DATA_DIR

    path = (data_dir or DATA_DIR) / "rca_database.json"
    return [RCA(**item) for item in _load_json(path)]


@lru_cache(maxsize=1)
def load_tsgs(data_dir: Path | None = None) -> list[TSG]:
    """Load all TSG records from the data directory."""
    from pf_recovery_agent.config import DATA_DIR

    path = (data_dir or DATA_DIR) / "tsg_database.json"
    return [TSG(**item) for item in _load_json(path)]
