from typing import Union, Optional, Callable
import numpy as np
import pandas as pd
import random
import math


class MyLineReg:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: Union[Callable[[float], float], float] = 0.1,
        weights: np.ndarray = [],
        metric: Optional[str] = None,
        reg: Optional[str] = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        sgd_sample: Optional[Union[float, int]] = None,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return f"MyLineReg class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def get_coef(self):
        return np.array(self.weights[1:])

    def get_best_score(self):
        return self.final_metric

    def _get_mse(self, y: pd.Series, y_hat: pd.Series):
        # calculate MSE
        mse_error = np.mean((y - y_hat) ** 2)
        if self.reg == "l1":
            mse_error = mse_error + self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            mse_error = mse_error + self.l2_coef * np.sum((self.weights) ** 2)
        elif self.reg == "elasticnet":
            mse_error = mse_error + self.l1_coef * np.sum(np.abs(self.weights))
            mse_error = mse_error + self.l2_coef * np.sum((self.weights) ** 2)

        return mse_error

    def _calculate_gradient(self, X: pd.DataFrame, y: pd.Series):
        def _sgn(weights: np.ndarray):
            return np.array([-1 if w < 0 else 1 if w > 0 else 0 for w in weights])

        # calculate gradient
        y_hat = X @ self.weights
        gradients = 2 / len(X) * (y_hat - y) @ X
        if self.reg == "l1":
            gradients = gradients + self.l1_coef * _sgn(self.weights)
        elif self.reg == "l2":
            gradients = gradients + self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            gradients = gradients + self.l1_coef * _sgn(self.weights)
            gradients = gradients + self.l2_coef * 2 * self.weights

        return gradients

    def _add_bias_column(self, X: pd.DataFrame):
        # insert
        X.insert(0, "x0", np.ones(len(X)))
        return X

    def _get_metric(self, X: pd.DataFrame, y: pd.Series):
        # return metric
        y_hat = X @ self.weights
        if self.metric is None:
            metric_val = None
        elif self.metric == "mae":
            metric_val = np.mean(np.abs(y - y_hat))
        elif self.metric == "mse":
            metric_val = np.mean((y - y_hat) ** 2)
        elif self.metric == "rmse":
            metric_val = np.mean((y - y_hat) ** 2)
            metric_val = np.sqrt(metric_val)
        elif self.metric == "mape":
            metric_val = 100 * np.mean(np.abs((y - y_hat) / y))
        elif self.metric == "r2":
            metric_val = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

        return metric_val

    def _fix_random_seed(self):
        random.seed(self.random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False):
        # fix random_seed
        self._fix_random_seed()
        # insert bias column
        X = self._add_bias_column(X)
        # create weights
        self.weights = np.ones(X.shape[1])
        # get initial mse
        mse_error = self._get_mse(y, X @ self.weights)
        # gradient descent
        for iter_num in range(1, self.n_iter + 1):
            # get minibatch
            if self.sgd_sample is None:
                x_batch, y_batch = X, y
            else:
                if isinstance(self.sgd_sample, int):
                    sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                elif isinstance(self.sgd_sample, float):
                    sample_rows_idx = random.sample(
                        range(X.shape[0]), math.ceil(self.sgd_sample * X.shape[0])
                    )
                x_batch, y_batch = X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]

            mse_error = self._get_mse(y_batch, x_batch @ self.weights)
            gradients = self._calculate_gradient(x_batch, y_batch)
            if callable(self.learning_rate):
                self.weights = self.weights - self.learning_rate(iter_num) * gradients
            else:
                self.weights = self.weights - self.learning_rate * gradients
            if verbose:
                if iter_num == 0:
                    metric_val = self._get_metric(x_batch, y_batch)
                    print(
                        f"start | loss: {round(mse_error, 3)} | {self.metric} : {metric_val}"
                    )
                elif iter_num % verbose == 0:
                    metric_val = self._get_metric(x_batch, y_batch)
                    print(
                        f"{iter_num} | loss: {round(mse_error, 3)} | {self.metric} : {metric_val}"
                    )

        # get final metric
        self.final_metric = self._get_metric(X, y)

    def predict(self, X: pd.DataFrame):
        # get predict
        X = self._add_bias_column(X)
        return X @ self.weights
