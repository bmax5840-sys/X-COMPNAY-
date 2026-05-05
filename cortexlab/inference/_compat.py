"""Optional pandas compatibility helpers.

The package uses pandas when it is installed, but keeps a tiny dataframe-like
fallback so the inference facade remains usable in minimal environments.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

try:  # pragma: no cover - exercised only when pandas is installed
    import pandas as pandas  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - default in the execution sandbox
    pandas = None


class SimpleDataFrame:
    """Small subset of ``pandas.DataFrame`` used by :mod:`predictor`.

    It supports column access, assignment, sorting, copying, ``empty``,
    ``columns``, ``len()``, and ``itertuples(index=False)``.  It is not intended
    to replace pandas for analytical workloads.
    """

    def __init__(
        self,
        data: Mapping[str, Iterable[Any]] | Iterable[Mapping[str, Any]] | None = None,
        *,
        columns: Sequence[str] | None = None,
    ) -> None:
        self._columns = list(columns or [])
        self._rows: list[dict[str, Any]] = []

        if data is None:
            return
        if isinstance(data, Mapping):
            self._from_mapping(data)
        else:
            self._from_records(data)

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def copy(self) -> "SimpleDataFrame":
        return SimpleDataFrame([row.copy() for row in self._rows], columns=self._columns)

    def sort_values(self, by: str) -> "SimpleDataFrame":
        return SimpleDataFrame(
            sorted(self._rows, key=lambda row: row[by]), columns=self._columns
        )

    def reset_index(self, drop: bool = False) -> "SimpleDataFrame":
        if drop:
            return self.copy()
        rows = []
        for index, row in enumerate(self._rows):
            indexed = {"index": index, **row}
            rows.append(indexed)
        return SimpleDataFrame(rows, columns=["index", *self._columns])

    def itertuples(self, index: bool = False):
        fields = (["Index"] if index else []) + self._columns
        Row = namedtuple("Row", fields)  # noqa: PYI024 - dynamic tuple shape
        for row_index, row in enumerate(self._rows):
            values = ([row_index] if index else []) + [row.get(column) for column in self._columns]
            yield Row(*values)

    def to_dict(self, orient: str = "dict") -> Any:
        if orient == "records":
            return [row.copy() for row in self._rows]
        if orient == "list":
            return {column: self[column] for column in self._columns}
        raise ValueError("SimpleDataFrame supports orient='records' or orient='list'")

    def __getitem__(self, column: str) -> list[Any]:
        return [row.get(column) for row in self._rows]

    def __setitem__(self, column: str, values: Iterable[Any] | Any) -> None:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            value_list = [values] * len(self._rows)
        else:
            value_list = list(values)
        if len(value_list) != len(self._rows):
            raise ValueError("Assigned column length must match row count")
        if column not in self._columns:
            self._columns.append(column)
        for row, value in zip(self._rows, value_list):
            row[column] = value

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"SimpleDataFrame({self._rows!r})"

    def _from_mapping(self, data: Mapping[str, Iterable[Any]]) -> None:
        columns = list(data.keys())
        raw_values: dict[str, Any] = dict(data)
        iterable_lengths = [
            len(value)
            for value in raw_values.values()
            if _is_column_iterable(value)
        ]
        row_count = max(iterable_lengths, default=1 if raw_values else 0)

        values: dict[str, list[Any]] = {}
        for column, value in raw_values.items():
            if _is_column_iterable(value):
                items = list(value)
                if len(items) != row_count:
                    raise ValueError(f"Column {column!r} has an inconsistent length")
                values[column] = items
            else:
                values[column] = [value] * row_count

        self._columns = columns if not self._columns else self._columns
        self._rows = [
            {column: values[column][index] for column in columns}
            for index in range(row_count)
        ]

    def _from_records(self, data: Iterable[Mapping[str, Any]]) -> None:
        rows = [dict(row) for row in data]
        if not self._columns:
            columns: list[str] = []
            for row in rows:
                for column in row:
                    if column not in columns:
                        columns.append(column)
            self._columns = columns
        self._rows = [{column: row.get(column) for column in self._columns} for row in rows]


def _is_column_iterable(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, Iterable)


def dataframe(data: Any = None, *, columns: Sequence[str] | None = None):
    if pandas is not None:  # pragma: no cover - depends on optional dependency
        return pandas.DataFrame(data, columns=columns)
    return SimpleDataFrame(data, columns=columns)


def is_dataframe(value: Any) -> bool:
    if isinstance(value, SimpleDataFrame):
        return True
    return pandas is not None and isinstance(value, pandas.DataFrame)


def to_numeric(values: Iterable[Any]) -> list[float | int]:
    converted: list[float | int] = []
    for value in values:
        numeric = float(value)
        converted.append(int(numeric) if numeric.is_integer() else numeric)
    return converted
