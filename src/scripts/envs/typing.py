from enum import StrEnum


class EnvType(StrEnum):
    BASELINE = "baseline"
    GREEDY = "greedy"
    SECURITY = "security"
    DENSE = "dense"
