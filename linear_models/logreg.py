import numpy as np
import pandas as pd
from typing import Union, Optional, Callable
from collections import defaultdict
import random
import math


class MyLogReg:
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
        self.eps = 1e-15
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_val = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def _fix_random_seed(self):
        random.seed(self.random_state)

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _add_bias_column(self, X: pd.DataFrame):
        # insert bis column
        X = pd.concat([pd.DataFrame(np.ones(len(X)), index=X.index), X], axis=1)
        return X

    def _sigmoid(self, vals: np.ndarray):
        # calculate sigmoid
        return 1 / (1 + np.exp(-vals))

    def _calculate_logloss(self, probs: np.ndarray, y: pd.Series):
        return -np.mean(
            y * np.log(probs + self.eps) + (1 - y) * np.log(probs + self.eps)
        )

    def _get_metric(self, X: pd.DataFrame, y: pd.Series):
        # get helping statistics
        vals = X @ self.weights
        probs = self._sigmoid(vals)
        y_hat = probs > 0.5
        tp, fp, tn, fn = 0, 0, 0, 0
        for label, pred in zip(y, y_hat):
            if label == pred == 1:
                tp += 1
            elif label == pred == 0:
                tn += 1
            elif label == 1:
                fn += 1
            elif label == 0:
                fp += 1
        # metrics
        if self.metric is None:
            metric_val = 0
        if self.metric == "accuracy":
            metric_val = (tp + tn) / (tp + tn + fp + fn)
        elif self.metric == "precision":
            metric_val = tp / (tp + fp)
        elif self.metric == "recall":
            metric_val = tp / (tp + fn)
        elif self.metric == "f1":
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            metric_val = 2 * (precision * recall) / (precision + recall)
        elif self.metric == "roc_auc":
            vals = list(zip(probs, y))
            vals = sorted(vals, key=lambda x: x[0], reverse=True)
            prob_prev, sum_ones = vals[0][0], 0
            sum_higher = 0
            prob2nums = defaultdict(int)
            sum_final = 0
            pos_nums, neg_nums = sum(y), len(y) - sum(y)
            for prob, label in vals:
                if prob < prob_prev:
                    sum_higher += sum_ones
                    sum_ones = 0

                if label == 1:
                    sum_ones += 1
                    prob2nums[prob] += 1
                elif label == 0:
                    same_prob = prob2nums[prob] / 2
                    higher_prob = sum_higher
                    sum_final += same_prob + higher_prob

            metric_val = 1 / (pos_nums * neg_nums) * sum_final

        return metric_val

    def _calculate_gradient(self, X: pd.DataFrame, y: pd.Series):
        def _sgn(weights: np.ndarray):
            return np.array([-1 if w < 0 else 1 if w > 0 else 0 for w in weights])

        # calculate gradient
        vals = X @ self.weights
        probs = self._sigmoid(vals)
        gradients = 1 / X.shape[0] * (probs - y) @ X
        if self.reg == "l1":
            gradients = gradients + self.l1_coef * _sgn(self.weights)
        elif self.reg == "l2":
            gradients = gradients + self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            gradients = gradients + self.l1_coef * _sgn(self.weights)
            gradients = gradients + self.l2_coef * 2 * self.weights

        return gradients

    def get_coef(self):
        return np.array(self.weights[1:])

    def predict_proba(self, X: pd.DataFrame):
        # predict classes
        X = self._add_bias_column(X)
        vals = X @ self.weights
        probs = self._sigmoid(vals)
        return probs

    def predict(self, X: pd.DataFrame):
        # predict probs
        probs = self.predict_proba(X)
        return probs > 0.5

    def get_best_score(self):
        return self.metric_val

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False):
        # fix random_seed
        self._fix_random_seed()
        # add bias column
        X = self._add_bias_column(X)
        # create weights
        self.weights = np.ones(X.shape[1])
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

            vals = x_batch @ self.weights
            probs = self._sigmoid(vals)
            loss = self._calculate_logloss(probs, y_batch)
            gradients = self._calculate_gradient(x_batch, y_batch)
            if callable(self.learning_rate):
                self.weights = self.weights - self.learning_rate(iter_num) * gradients
            else:
                self.weights = self.weights - self.learning_rate * gradients

            if verbose and iter_num == 1:
                self.metric_val = self._get_metric(x_batch, y_batch)
                print("start | loss: {loss} | {self.metric}: {self.metric_val}")
            elif verbose and iter_num % verbose == 0:
                self.metric_val = self._get_metric(x_batch, y_batch)
                print(f"{iter_num} | loss: {loss} | {self.metric}: {self.metric_val}")

        self.metric_val = self._get_metric(X, y)
