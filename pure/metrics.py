"""Valid metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime

import pandas as pd
import pyspark.sql as ps


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count

        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        mask = df[self.columns[0]].isna()
        for column in self.columns[1:]:
            if self.aggregation == "any":
                mask |= df[column].isna()
            elif self.aggregation == "all":
                mask &= df[column].isna()
            else:
                raise ValueError("Unknown value for aggregation")
        k = sum(mask)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count, when, isnan

        n = df.count()
        column = self.columns[0]
        mask = col(column).isNull() | isnan(col(column))
        for column in self.columns[1:]:
            if self.aggregation == "any":
                mask |= col(column).isNull() | isnan(col(column))
            elif self.aggregation == "all":
                mask &= col(column).isNull() | isnan(col(column))
            else:
                raise ValueError("Unknown value for aggregation")

        k = df.select(count(when(mask, column))).collect()[0][0]

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        m = len(df.drop_duplicates(subset=self.columns))
        k = n - m
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import countDistinct

        n = df.count()
        m = df.select(countDistinct(*self.columns)).collect()[0][0]
        k = n - m
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col

        n = df.count()
        k = df.filter(col(self.column) == self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column] < self.value)
        else:
            k = sum(df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col

        n = df.count()
        if self.strict:
            k = df.filter(col(self.column) < self.value).count()
        else:
            k = df.filter(col(self.column) <= self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x] < df[self.column_y])
        else:
            k = sum(df[self.column_x] <= df[self.column_y])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, isnan

        n = df.count()

        mask = isnan(col(self.column_x)) | col(self.column_x).isNull()
        mask |= isnan(col(self.column_y)) | col(self.column_y).isNull()
        df = df.filter(~(mask))

        if self.strict:
            k = df.filter(col(self.column_x) < col(self.column_y)).count()
        else:
            k = df.filter(col(self.column_x) <= col(self.column_y)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        ratio = df[self.column_x] / df[self.column_y]
        if self.strict:
            k = sum(ratio < df[self.column_z])
        else:
            k = sum(ratio <= df[self.column_z])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, isnan

        n = df.count()

        mask = isnan(col(self.column_x)) | col(self.column_x).isNull()
        mask |= isnan(col(self.column_y)) | col(self.column_y).isNull()
        mask |= isnan(col(self.column_z)) | col(self.column_z).isNull()
        df = df.filter(~(mask))

        ratio = col(self.column_x) / col(self.column_y)
        if self.strict:
            k = df.filter(ratio < col(self.column_z)).count()
        else:
            k = df.filter(ratio <= col(self.column_z)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        alpha = 1 - self.conf
        lcb = df[self.column].quantile(alpha / 2)
        ucb = df[self.column].quantile(alpha / 2 + self.conf)
        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql import DataFrameStatFunctions

        alpha = 1 - self.conf
        st = DataFrameStatFunctions(df)
        ci = st.approxQuantile(self.column, [alpha / 2, self.conf + alpha / 2], 0)
        return {"lcb": ci[0], "ucb": ci[1]}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        a = datetime.datetime.now()
        b = pd.to_datetime(df[self.column]).max()
        lag = (a - b).days
        a = a.strftime(self.fmt)
        b = b.strftime(self.fmt)
        return {"today": a, "last_day": b, "lag": lag}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, max

        a = datetime.datetime.now()
        b = df.select(max(col(self.column))).collect()[0][0]
        b = datetime.datetime.strptime(b, "%Y-%m-%d")

        lag = (a - b).days
        a = a.strftime(self.fmt)
        b = b.strftime(self.fmt)
        return {"today": a, "last_day": b, "lag": lag}