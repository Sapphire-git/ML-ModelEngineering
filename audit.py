"""Check the UCI Bike Sharing hourly data before choosing temporal splits."""

import argparse
import json

import pandas as pd


def load_hourly(path):
    raw = pd.read_csv(path, usecols=["dteday", "hr", "cnt"])
    dates = pd.to_datetime(raw["dteday"], format="%Y-%m-%d", errors="raise")
    hours = pd.to_numeric(raw["hr"], errors="raise")
    counts = pd.to_numeric(raw["cnt"], errors="raise")
    if raw.empty or dates.isna().any():
        raise ValueError("The file must contain rows with valid dates.")
    if not (hours.between(0, 23) & hours.mod(1).eq(0)).all():
        raise ValueError("Every hour must be an integer from 0 to 23.")
    if not (counts.ge(0) & counts.mod(1).eq(0)).all():
        raise ValueError("Rental counts must be nonnegative integers.")
    timestamps = dates + pd.to_timedelta(hours, unit="h")
    return pd.Series(counts.to_numpy(), index=timestamps, name="cnt").sort_index()


def audit(counts):
    unique_hours = counts.index.unique()
    timeline = pd.date_range(unique_hours.min(), unique_hours.max(), freq="h")
    missing = timeline.difference(unique_hours)
    available_lags = (unique_hours - pd.Timedelta(days=7)).isin(unique_hours)
    return {
        "rows": len(counts),
        "first_hour": str(unique_hours.min()),
        "last_hour": str(unique_hours.max()),
        "duplicate_rows": len(counts) - len(unique_hours),
        "missing_hours": len(missing),
        "missing_hours_by_hour_of_day": missing.hour.value_counts().sort_index().to_dict(),
        "unique_hours_without_weekly_lag": int((~available_lags).sum()),
        "zero_counts": int(counts.eq(0).sum()),
        "median_count": float(counts.median()),
        "p95_count": float(counts.quantile(0.95)),
        "maximum_count": int(counts.max()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to UCI Bike Sharing hour.csv")
    args = parser.parse_args()
    print(json.dumps(audit(load_hourly(args.csv)), indent=2))
