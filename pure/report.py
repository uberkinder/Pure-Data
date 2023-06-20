"""Valid report."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyspark.sql as ps

from pure.metrics import Metric
from pure.sql_connector import ClickHouseConnector

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """Valid report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame, str]], conn: List[str] = None) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables)

        if self.engine == "clickhouse":
            return self._fit_clickhouse(tables, conn)

        if self.engine == "postgresql":
            return self._fit_postgresql(tables, conn)

        if self.engine == "mssql":
            return self._fit_mssql(tables, conn)

        raise NotImplementedError("Only pandas and pyspark APIs currently supported!")

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: Pandas"""
        if not hasattr(self, "_reports"):
            self._reports = {}
        hash_key = hash(str(tables))
        if hash_key in self._reports:
            return self._reports[hash_key]

        self.report_ = {}
        report = self.report_

        # Run check-by-check
        rows = []
        for table, metric, limits in self.checklist:
            # Init resulting row
            row = {
                "table_name": table,
                "metric_name": metric.__class__.__name__,
                "limits": str(limits),
            }

            # Run check
            try:
                # Run metric
                df = tables[table]
                values = metric(self.engine, df)
                # row["values"] = values
                row["status"] = "."
                row["error"] = ""
                limit_values = {}
                if limits:
                    # Check metrics
                    for key, (a, b) in limits.items():
                        value = values[key]
                        limit_values[key] = np.around(value, 3)
                        if not (a <= value <= b):
                            row["status"] = "F"
                row["values"] = limit_values
                row["metric_values"] = values
            except Exception as e:
                row["status"] = "E"
                row["error"] = type(e).__name__
                row["values"] = {}
                row["metric_values"] = {}
            row["metric_params"] = list(vars(metric).values())
            # Append to other rows
            rows.append(row)

        # Print the results
        tables = sorted(list(set(tables.keys())))
        result = pd.DataFrame(rows)
        order = [
            "table_name",
            "metric_name",
            "limits",
            "values",
            "status",
            "error",
            "metric_values",
            "metric_params",
        ]
        result = result[order]

        report["title"] = f"DQ Report for tables {tables}"
        report["result"] = result

        # Print the statistics
        total = len(result)
        passed = sum(result["status"] == ".")
        failed = sum(result["status"] == "F")
        errors = sum(result["status"] == "E")

        report["passed"] = passed
        report["passed_pct"] = round(100 * passed / total, 2)
        report["failed"] = failed
        report["failed_pct"] = round(100 * failed / total, 2)
        report["errors"] = errors
        report["errors_pct"] = round(100 * errors / total, 2)
        report["total"] = total

        self._reports[hash_key] = report

        return report

    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""

        self.report_ = {}
        report = self.report_

        # Run check-by-check
        rows = []
        for table, metric, limits in self.checklist:
            # Init resulting row
            row = {
                "table_name": table,
                "metric_name": metric.__class__.__name__,
                "limits": str(limits),
            }

            # Run check
            try:
                # Run metric
                df = tables[table]
                values = metric(self.engine, df)
                row["status"] = "."
                row["error"] = ""
                limit_values = {}
                if limits:
                    # Check metrics
                    for key, (a, b) in limits.items():
                        value = values[key]
                        limit_values[key] = np.around(value, 3)
                        if not (a <= value <= b):
                            row["status"] = "F"
                row["values"] = limit_values
                row["metric_values"] = values
            except Exception as e:
                row["status"] = "E"
                row["error"] = type(e).__name__
                row["values"] = {}
                row["metric_values"] = {}
            row["metric_params"] = list(vars(metric).values())
            # Append to other rows
            rows.append(row)

        # Print the results
        tables = sorted(list(set(tables.keys())))
        result = pd.DataFrame(rows)
        order = [
            "table_name",
            "metric_name",
            "limits",
            "values",
            "status",
            "error",
            "metric_values",
            "metric_params",
        ]
        result = result[order]

        report["title"] = f"DQ Report for tables {tables}"
        report["result"] = result

        # Print the statistics
        total = len(result)
        passed = sum(result["status"] == ".")
        failed = sum(result["status"] == "F")
        errors = sum(result["status"] == "E")

        report["passed"] = passed
        report["passed_pct"] = round(100 * passed / total, 2)
        report["failed"] = failed
        report["failed_pct"] = round(100 * failed / total, 2)
        report["errors"] = errors
        report["errors_pct"] = round(100 * errors / total, 2)
        report["total"] = total

        return report

    def _fit_clickhouse(self, tables: Dict[str, str], conn : List[str]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: ClickHouse"""
        # Create database connection
        sql_client = ClickHouseConnector(*conn)

        self.report_ = {}
        report = self.report_

        # Run check-by-check
        rows = []
        for table, metric, limits in self.checklist:
            # Init resulting row
            row = {
                "table_name": table,
                "metric_name": metric.__class__.__name__,
                "limits": str(limits),
            }

            # Run check
            try:
                # Run metric
                df = tables[table]
                values = metric(self.engine, df, sql_client)
                row["status"] = "."
                row["error"] = ""
                limit_values = {}
                if limits:
                    # Check metrics
                    for key, (a, b) in limits.items():
                        value = values[key]
                        limit_values[key] = np.around(value, 3)
                        if not (a <= value <= b):
                            row["status"] = "F"
                row["values"] = limit_values
                row["metric_values"] = values
            except Exception as e:
                row["status"] = "E"
                row["error"] = type(e).__name__
                row["values"] = {}
                row["metric_values"] = {}
            row["metric_params"] = list(vars(metric).values())
            # Append to other rows
            rows.append(row)

        # Print the results
        tables = sorted(list(set(tables.keys())))
        result = pd.DataFrame(rows)
        order = [
            "table_name",
            "metric_name",
            "limits",
            "values",
            "status",
            "error",
            "metric_values",
            "metric_params",
        ]
        result = result[order]

        report["title"] = f"DQ Report for tables {tables}"
        report["result"] = result

        # Print the statistics
        total = len(result)
        passed = sum(result["status"] == ".")
        failed = sum(result["status"] == "F")
        errors = sum(result["status"] == "E")

        report["passed"] = passed
        report["passed_pct"] = round(100 * passed / total, 2)
        report["failed"] = failed
        report["failed_pct"] = round(100 * failed / total, 2)
        report["errors"] = errors
        report["errors_pct"] = round(100 * errors / total, 2)
        report["total"] = total

        return report
    def _fit_postgresql(self, tables: Dict[str, str]) -> Dict:
        pass

    def _fit_mssql(self, tables: Dict[str, str]) -> Dict:
        pass

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
