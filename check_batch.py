"""Exercise batch inference against valid, missing-lag and malformed records."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from forecast import batch


def main():
    with TemporaryDirectory(dir=".") as folder:
        folder = Path(folder)
        history = folder / "history.csv"
        requests = folder / "requests.csv"
        output = folder / "output.csv"
        history.write_text("dteday,hr,cnt\n2012-08-25,9,1000000\n", encoding="utf-8")
        requests.write_text(
            "record_id,forecast_hour\n001,2012-09-01 09:00\n002,2012-09-01 10:00\n",
            encoding="utf-8",
        )
        batch(requests, history, "model.joblib", output)
        scored = pd.read_csv(output, dtype={"record_id": "string"})
        assert scored["record_id"].tolist() == ["001", "002"]
        assert scored["prediction"].ge(0).all()
        assert scored["method"].eq("model").all()
        assert scored["lower_90"].le(scored["prediction"]).all()
        assert scored["upper_90"].ge(scored["prediction"]).all()

        requests.write_text("record_id,forecast_hour\n001,2012-09-01 09:15\n", encoding="utf-8")
        try:
            batch(requests, history, "model.joblib", output)
        except ValueError as error:
            assert "whole hours" in str(error)
        else:
            raise AssertionError("Accepted a timestamp that was not on the hour.")
    print("Batch checks passed for missing and extreme lags, preserved IDs, and malformed hours.")


if __name__ == "__main__":
    main()
