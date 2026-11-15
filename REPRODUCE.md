# Day-ahead bike rental forecasting

Use Python 3.12. Install with `python -m pip install -r requirements.txt`.
Use the original hourly CSV already present at the repository root. Its documentation is in README.md. [Dataset and license](https://doi.org/10.24432/C5W894)

```text
python audit.py hour.csv
python forecast.py train hour.csv model.joblib results.json
python forecast.py benchmark hour.csv model.joblib
python check_batch.py
python forecast.py batch requests.csv history.csv model.joblib forecasts.csv
```

Use only trusted artifacts with joblib. A request uses `record_id,forecast_hour`
with timestamps such as `2012-09-01 09:00`. Use one forecast date per file.
History uses the source CSV schema with `dteday,hr,cnt` and contains only observations
prior to the forecast date. Use consistent local clock hours for both files. The output
preserves IDs and includes point predictions, interval bounds where available, and
the scoring method. Invalid inputs return a nonzero exit code.

The benchmark replays held-out days through the in-memory scoring path. It times
history filtering, feature creation, prediction and construction of output records.
It excludes CSV and model loading and CSV writing.

The checked-in JSON files record the measured experiment results. Re-run the commands to regenerate them.
