"""
This primitive an implementation of Amazon's Chronos2 model for timeseries forecasting.

The model implementation can be found at
https://huggingface.co/amazon/chronos-2

Note: This primitive assumes that Chronos2 doesn't care about specific timestamps
of the data. We fill in the timestamps with a linear sequence of timestamps in order
for the model to work.
"""

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


class Chronos2:
    """Chronos2 model for timeseries forecasting.

    Args:
        pred_len (int):
            Prediction horizon length. Default to 1.
        repo_id (str):
            Directory of the model checkpoint. Default to "amazon/chronos-2"
        batch_size(int):
            Size of one batch. Default to 32.
        target (int):
            Index of target column in multivariate case. Default to 0.
        start_time (datetime):
            Start time of the timeseries. Default to Jan 1, 2020 00:00:00.
        time_interval (int):
            Time interval between two samples in seconds. Default to 600.
    """

    def __init__(self,
                 pred_len=1,
                 repo_id="amazon/chronos-2",
                 batch_size=32,
                 target=0,
                 start_time=pd.to_datetime("2000-01-01 00:00:00"),
                 time_interval=600):

        self.pred_len = pred_len
        self.batch_size = batch_size
        self.target = f"{target}"
        self.start_time = start_time
        self.time_interval = pd.Timedelta(seconds=time_interval)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Chronos2Pipeline.from_pretrained(repo_id, device_map=device)

    def predict(self, X, force=False):
        """Forecasting timeseries

        Args:
            X (ndarray):
                input timeseries with shape (n_windows, window_size, n_features).
        Return:
            ndarray:
                forecasted timeseries.
        """
        n_windows, window_size, n_features = X.shape

        outs = []

        for i in range(0, n_windows, self.batch_size):
            x_batch = self.convert_to_df(X[i:i + self.batch_size], start_batch_at=i)
            y_batch = self.model.predict_df(
                df=x_batch,
                prediction_length=self.pred_len,
                quantile_levels=[0.5],
                id_column="item_id",
                timestamp_column="timestamp",
                target=self.target,
            )

            y_batch = y_batch.sort_values(["item_id", "timestamp"])
            preds = np.stack(
                y_batch.groupby("item_id", sort=False)["predictions"]
                .apply(lambda s: s.to_numpy())
                .to_list()
            )
            outs.append(preds)

        return np.concatenate(outs, axis=0)

    def convert_to_df(self, x_batch, start_batch_at=0):
        n_windows_in_batch, window_size, n_features = x_batch.shape

        rows = []
        for window in range(n_windows_in_batch):
            for data_entry in range(window_size):
                rows.append({
                    "timestamp": self.start_time + self.time_interval * data_entry,
                    "item_id": f"window_{start_batch_at + window}",
                    **{f"{i}": x_batch[window, data_entry, i] for i in range(n_features)}
                })

        rows = pd.DataFrame(rows)
        return rows
