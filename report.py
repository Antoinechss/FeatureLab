import datetime
import json
import pandas as pd

test_dset_path = "test_dset.csv"
test_dset = pd.read_csv(test_dset_path)


def report_column(df: pd.DataFrame, column_id):

    column = df[column_id]
    report = {}

    report["name"] = column_id
    report["pandas_dtype"] = str(column.dtype)
    report["missing_pct"] = float(column.isnull().sum() / len(column))
    report["unique_count"] = int(column.nunique())
    report["unique_ratio"] = float(column.nunique() / len(column))

    if column.dtype in ["int64", "float64"]:
        report["stats"] = {
            "mean": float(column.mean()),
            "std": float(column.std()),
            "min": float(column.min()),
            "max": float(column.max()),
            "skewness": float(column.skew()),
        }

    if pd.api.types.is_datetime64_any_dtype(column):
        report["datetime_range"] = {
            "min": str(column.min()),
            "max": str(column.max()),
        }

    if column.dtype == "object":
        report["avg_length"] = float(column.dropna().astype(str).str.len().mean())

    return report


def report(df):
    report = {}

    report["meta"] = {
        "schema_version": "1.0",
        "generated_at": datetime.datetime.now(),
        "profile_config": {
            "skew_threshold": 2.0,
            "high_cardinality_unique_ratio": 0.2,
            "correlation_threshold": 0.9,
        },
    }

    # Overview
    report["overview"] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_pct": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "dtype_distribution": {str(k): int(v) for k, v in df.dtypes.value_counts().items()}
    }

    # Columns
    report["columns"] = []
    for column_name in df.columns:
        report["columns"].append(report_column(df, column_name))

    return report


print(json.dumps(report(test_dset), indent=4, default=str))
