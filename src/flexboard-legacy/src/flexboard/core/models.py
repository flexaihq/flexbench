import typing as t
from dataclasses import dataclass, field


@dataclass
class Column:
    name: str
    type: t.Literal["Numerical", "Categorical"]
    is_feature: bool = False
    enabled: bool = field(default=True, compare=False)


@dataclass
class Feature:
    type: t.Literal["Numerical", "Categorical"] = "Numerical"
    col_a: str | None = None
    col_b: str | None = None
    operator: t.Literal["add", "sub", "mul", "truediv", "concat"] | None = None
    name: str = ""
    enabled: bool = field(default=True, compare=False)


@dataclass
class Filter:
    type: t.Literal["Numerical", "Categorical"]
    column: str
    range: tuple[int | float, int | float] | None
    values: list[str] | None
    enabled: bool = field(default=True, compare=False)


@dataclass
class BenchmarkState:
    features: list[Feature]
    filters: list[Filter]
    columns: dict[str, Column]
