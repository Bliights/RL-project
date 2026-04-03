from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BenchmarkConfig:
    env_id: str
    env_config: dict[str, Any]
