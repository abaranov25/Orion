"""
This primitive an implementation of Cisco's Time Series Foundation Model for timeseries forecasting.

The model implementation can be found at
https://arxiv.org/pdf/2511.19841

We use code from https://github.com/splunk/cisco-time-series-model
in this primitive, which can be found in the relative import path.
"""

import torch
import numpy as np

try:
    from ..pipelines.pretrained.cisco.cisco_modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint
except ImportError as ie:
    ie.msg += (
        '\n\nIt seems like `cisco` cannot be imported.\n'
        'It is likely that relative import is failing. Please flag this issue. \n'
    )
    raise


class Cisco:
    """Cisco model for timeseries forecasting.

    Args:
        window_size (int):
            Window size of each sample. Default to 256.
        step (int):
            Stride length between samples. Default to 1.
        pred_len (int):
            Prediction horizon length. Default to 1.
        repo_id (str):
            Directory of the model checkpoint. Default to "cisco-ai/cisco-time-series-model-1.0-preview"
        batch_size(int):
            Size of one batch. Default to 32.
        freq (int):
            Frequency. TimesFM expects a categorical indicator valued in {0, 1, 2}.
            Default to 0.
        target (int):
            Index of target column in multivariate case. Default to 0.
        start_time (datetime):
            Start time of the timeseries. Default to Jan 1, 2020 00:00:00.
        time_interval (int):
            Time interval between two samples in seconds. Default to 600.
    """

    def __init__(
        self,
        window_size=30720, # note that cisco expects a large window size because it uses long term context
        pred_len=1,
        repo_id="cisco-ai/cisco-time-series-model-1.0-preview",
        batch_size=32,
        target=0,
        return_quantile=None,
    ):
        self.window_size = int(window_size)
        self.pred_len = int(pred_len)
        self.batch_size = int(batch_size)
        self.target = int(target)
        self.return_quantile = return_quantile

        # Match the model-card example
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        hparams = TimesFmHparams(
            num_layers=50,
            use_positional_embedding=False,
            backend=backend,
        )
        ckpt = TimesFmCheckpoint(huggingface_repo_id=repo_id)

        self.model = CiscoTsmMR(
            hparams=hparams,
            checkpoint=ckpt,
            use_resolution_embeddings=True,
            use_special_token=True,
        )

    def predict(self, X):
        """Forecast.

        Args:
            X (ndarray): shape (n_windows, window_size, n_features)
        Returns:
            ndarray: shape (n_windows, pred_len)
        """

        n_windows = X.shape[0]

        outs = []
        for i in range(0, n_windows, self.batch_size):
            x_batch = X[i:i + self.batch_size, :self.window_size, self.target].astype(np.float32)

            series_list = [x_batch[j] for j in range(x_batch.shape[0])] # x_batch.shape[0] could be lower than self.batch_size
            forecast_list = self.model.forecast(series_list, horizon_len=self.pred_len)
            preds = np.stack([f["mean"] for f in forecast_list], axis=0)

            outs.append(preds)

        return np.concatenate(outs, axis=0)




if __name__ == "__main__":
    cisco_predictor = CiscoPredictor(window_size=256, pred_len=16, batch_size=32, target=0, return_quantile=None)
    X = np.random.rand(100, 256, 10).astype(np.float32)
    y = cisco_predictor.predict(X)
    print(y.shape) # should be (100, 16)
    print(y[:2])