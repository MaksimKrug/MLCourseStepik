import pandas as pd
import numpy as np


class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train_size = (0, 0)
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _weighted_pred(self, dist: np.ndarray):
        # sort by distances
        groups = sorted(zip(self.y, dist), key=lambda x: x[1])
        groups = groups[: self.k]

        if self.weight == "rank":
            denominator = sum(1 / i for i in range(1, len(groups) + 1))
            weights = np.array(
                [(1 / idx) / denominator for idx in range(1, len(groups) + 1)]
            )
        elif self.weight == "distance":
            denominator = sum(1 / d[1] for d in groups)
            weights = np.array([(1 / d[1]) / denominator for d in groups])

        return np.sum([groups[idx][0] * weights[idx] for idx in range(len(groups))])

    def _get_distance(self, X_pred: pd.DataFrame):
        x_values, pred_values = self.X.values, X_pred.values
        if self.metric == "euclidean":
            dist = (pred_values[:, np.newaxis, :] - x_values[np.newaxis, :, :]) ** 2
            dist = np.sum(dist, axis=2)
            dist = np.sqrt(dist)
        elif self.metric == "chebyshev":
            dist = np.abs(pred_values[:, np.newaxis, :] - x_values[np.newaxis, :, :])
            dist = dist.max(axis=2)
        elif self.metric == "manhattan":
            dist = np.abs(pred_values[:, np.newaxis, :] - x_values[np.newaxis, :, :])
            dist = np.sum(dist, axis=2)
        elif self.metric == "cosine":
            numerator = pred_values[:, np.newaxis, :] * x_values[np.newaxis, :, :]
            numerator = np.sum(numerator, axis=2)
            den1 = np.sqrt(np.sum(pred_values**2, axis=1))
            den2 = np.sqrt(np.sum(x_values**2, axis=1))
            denominator = np.outer(den1, den2)
            dist = 1 - (numerator / denominator)

        return dist

    def _custom_mode(self, array: np.ndarray):
        unique, counts = np.unique(array, return_counts=True)
        max_count_index = np.argmax(counts)
        mode_val = unique[max_count_index]
        if len(counts) == 1:
            return mode_val
        elif counts[0] > counts[1]:
            return 0
        else:
            return 1

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = tuple(X.shape)

    def predict(self, X_pred: pd.DataFrame):
        # get distances
        distances = self._get_distance(X_pred)
        # predicts
        if self.weight == "uniform":
            top_indices = np.argsort(distances, axis=1)[..., : self.k]
            values = self.y.values[top_indices]
            predicts = np.mean(values, axis=1)
        elif self.weight in ["rank", "distance"]:
            predicts = np.apply_along_axis(self._weighted_pred, axis=1, arr=(distances))

        return predicts
