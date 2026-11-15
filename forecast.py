"""Train, evaluate, and batch-score day-ahead bike rental forecasts."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from audit import load_hourly


def features(hours, history):
    previous_week = history.reindex(hours - pd.Timedelta(days=7)).to_numpy()
    return pd.DataFrame({
        "hour": hours.hour,
        "weekday": hours.dayofweek,
        "month": hours.month,
        "weekly_lag": previous_week,
    }, index=hours)


def baseline(frame, medians):
    return frame["weekly_lag"].fillna(frame["hour"].map(medians)).to_numpy()


def predict(model, frame):
    predictions = np.maximum(model.predict(frame), 0)
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned non-finite forecasts.")
    return predictions


def metrics(truth, prediction):
    errors = np.abs(truth.to_numpy() - prediction)
    return {"mae": float(errors.mean()), "wape": float(errors.sum() / truth.sum())}


def read_history(path):
    history = load_hourly(path)
    if not history.index.is_unique:
        raise ValueError("History contains duplicate hours.")
    return history


def train(data, artifact, results):
    counts = read_history(data)
    frame = features(counts.index, counts)
    fold_dates = [
        ("2012-01-01", "2012-03-01"),
        ("2012-03-01", "2012-05-01"),
        ("2012-05-01", "2012-07-01"),
    ]
    validation = {}
    for leaves in (15, 31):
        validation[leaves] = []
        for start, stop in fold_dates:
            fit = frame.index < start
            check = (frame.index >= start) & (frame.index < stop)
            model = HistGradientBoostingRegressor(max_leaf_nodes=leaves, random_state=42)
            model.fit(frame.loc[fit], counts.loc[fit])
            validation[leaves].append(metrics(counts.loc[check], predict(model, frame.loc[check])))
    best = min(validation, key=lambda leaves: np.mean([m["mae"] for m in validation[leaves]]))
    fit = frame.index < "2012-07-01"
    calibrate = (frame.index >= "2012-07-01") & (frame.index < "2012-09-01")
    test = frame.index >= "2012-09-01"
    model = HistGradientBoostingRegressor(max_leaf_nodes=best, random_state=42)
    model.fit(frame.loc[fit], counts.loc[fit])
    medians = counts.loc[fit].groupby(counts.index[fit].hour).median().to_dict()
    default = float(counts.loc[fit].median())
    medians = {hour: medians.get(hour, default) for hour in range(24)}
    residuals = np.abs(counts.loc[calibrate].to_numpy() - predict(model, frame.loc[calibrate]))
    radius = float(np.quantile(residuals, 0.90, method="higher"))
    predicted = predict(model, frame.loc[test])
    lower = np.maximum(predicted - radius, 0)
    upper = predicted + radius
    summary = {
        "rolling_validation": validation,
        "chosen_leaf_limit": best,
        "fit_rows": int(fit.sum()),
        "calibration_rows": int(calibrate.sum()),
        "test_rows": int(test.sum()),
        "test_baseline": metrics(counts.loc[test], baseline(frame.loc[test], medians)),
        "test_model": metrics(counts.loc[test], predicted),
        "interval_radius": radius,
        "interval_coverage": float(((counts.loc[test] >= lower) & (counts.loc[test] <= upper)).mean()),
        "mean_interval_width": float((upper - lower).mean()),
    }
    joblib.dump({"model": model, "medians": medians, "radius": radius}, artifact)
    Path(results).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def score(frame, bundle):
    try:
        values = predict(bundle["model"], frame)
    except (AttributeError, ValueError, RuntimeError):
        logging.exception("Model scoring failed. Using the seasonal baseline.")
        return pd.DataFrame({
            "prediction": baseline(frame, bundle["medians"]),
            "lower_90": np.nan,
            "upper_90": np.nan,
            "method": "baseline",
        }, index=frame.index)
    radius = bundle["radius"]
    return pd.DataFrame({
        "prediction": values,
        "lower_90": np.maximum(values - radius, 0),
        "upper_90": values + radius,
        "method": "model",
    }, index=frame.index)


def batch(requests, history, artifact, output):
    data = pd.read_csv(requests, dtype={"record_id": "string"})
    hours = pd.DatetimeIndex(pd.to_datetime(data["forecast_hour"], format="%Y-%m-%d %H:%M",
                                             errors="raise"))
    ids = data["record_id"]
    if data.empty or ids.isna().any() or not ids.is_unique:
        raise ValueError("Requests require unique, nonmissing record IDs.")
    if hours.hasnans or (hours != hours.floor("h")).any():
        raise ValueError("Forecast timestamps must identify whole hours.")
    if hours.normalize().nunique() != 1:
        raise ValueError("A batch must contain hours from exactly one forecast day.")
    history = read_history(history)
    if history.index.max() >= hours.min().normalize():
        raise ValueError("History must end before the forecast day.")
    bundle = joblib.load(artifact)
    forecasts = score(features(hours, history), bundle)
    forecasts.insert(0, "forecast_hour", hours)
    forecasts.insert(0, "record_id", ids.to_numpy())
    forecasts.to_csv(output, index=False)
    logging.info("Wrote %d forecasts to %s", len(forecasts), output)


def benchmark(data, artifact):
    counts = read_history(data)
    hours = counts.index[counts.index >= "2012-09-01"]
    days = [group.index for _, group in counts.reindex(hours).groupby(hours.normalize())]
    bundle = joblib.load(artifact)
    score(features(days[0], counts), bundle)
    elapsed = []
    for _ in range(10):
        start = perf_counter()
        for day in days:
            available = counts.loc[counts.index < day[0].normalize()]
            score(features(day, available), bundle)
        elapsed.append(perf_counter() - start)
    print(json.dumps({
        "records_per_run": len(hours),
        "daily_batches_per_run": len(days),
        "runs": len(elapsed),
        "median_seconds": float(np.median(elapsed)),
        "median_records_per_second": float(np.median(len(hours) / np.asarray(elapsed))),
    }, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fields in {
        "train": ("data", "artifact", "results"),
        "batch": ("requests", "history", "artifact", "output"),
        "benchmark": ("data", "artifact"),
    }.items():
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        {"train": train, "batch": batch, "benchmark": benchmark}[command](**args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
