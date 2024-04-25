import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Union, Optional, Callable
from collections import defaultdict

import random


class MyLineReg:
    def __init__(
        self,
        n_iter: int = 100,
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

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int] = False):
        # fix random_seed
        random.seed(self.random_state)
        # insert bias column
        X.insert(0, "bias", 1)
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
                        range(X.shape[0]), round(self.sgd_sample * X.shape[0])
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
        X.insert(0, "bias", 1)
        return X @ self.weights


class Node:
    def __init__(
        self,
        col: str,
        val: float,
        criterion_val: float = 0,
        elements: int = 0,
        left=None,
        right=None,
    ):
        self.col = col
        self.val = val
        self.left = left
        self.right = right
        self.criterion_val = criterion_val
        self.elements = elements


class Leaf:
    def __init__(
        self,
        name: str = "leaf_left",
        pred: float = 0,
        criterion_val: float = 0,
        elements: int = 0,
    ):
        self.name = name
        self.pred = pred
        self.criterion_val = criterion_val
        self.elements = elements


class MyTreeReg:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
        criterion: str = "mse",
        examples_num: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(max_leafs, 2)
        self.bins = bins
        self.criterion = criterion

        self.leafs_cnt = 0
        self.examples_num = examples_num
        self.required_leafs = 0
        self.tree = {}
        self.delimeters = {}
        self.fi = {}

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _calculate_criterion(self, y: np.ndarray):
        if len(y) == 0:
            return 0
        return np.mean((y - y.mean()) ** 2)

    def _calculate_information_gain(
        self, vals: np.ndarray, left: np.ndarray, right: np.ndarray
    ):
        # calculate information criterions
        s_0 = self._calculate_criterion(vals[:, 1])
        s_1 = self._calculate_criterion(left[:, 1])
        s_2 = self._calculate_criterion(right[:, 1])

        # calculate inofrmation gain
        ig = s_0 - len(left) / len(vals) * s_1 - len(right) / len(vals) * s_2

        return ig

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_split = (X.columns[0], 0, 0)
        for col_name in X.columns:
            # sort vals
            vals = np.stack((X[col_name].values, y.values), axis=1)
            sorted_indices = np.argsort(vals[:, 0])
            vals = vals[sorted_indices]
            # get delimeters
            delimeters = self.delimeters[col_name]

            for delimeter_value in delimeters:
                left = vals[vals[:, 0] <= delimeter_value]
                right = vals[vals[:, 0] > delimeter_value]
                split_ig = self._calculate_information_gain(vals, left, right)
                if split_ig > best_split[2]:
                    best_split = (col_name, delimeter_value, split_ig)

        return best_split

    def _split(self, X: pd.DataFrame, y: pd.Series, col: str, val: float):
        df = pd.concat((X, y), axis=1)
        # split
        left = df.loc[df[col] <= val]
        right = df.loc[df[col] > val]
        # unpack
        left_X, left_y = left.iloc[:, :-1], left.iloc[:, -1]
        right_X, right_y = right.iloc[:, :-1], right.iloc[:, -1]

        return (left_X, left_y), (right_X, right_y)

    def _add_leaf(self, node: Node, y: pd.Series, direction: str):
        criterion_val = self._calculate_criterion(y.values)

        if direction == "left":
            node.left = Leaf("leaf_left", y.mean(), criterion_val, len(y))
            self.leafs_cnt += 1
            self.required_leafs -= 1
        elif direction == "right":
            node.right = Leaf("leaf_right", y.mean(), criterion_val, len(y))
            self.leafs_cnt += 1
            self.required_leafs -= 1

    def _check_delimeters_existence(self, X: pd.DataFrame):
        for col_name in X.columns:
            vals = X[col_name].values
            delimeters = self.delimeters[col_name]
            if any([vals.min() <= delimeter <= vals.max() for delimeter in delimeters]):
                return True
        return False

    def _dfs(self, X: pd.DataFrame, y: pd.Series, node: Node, current_depth: int):
        # split by parrent
        (X_left, y_left), (X_right, y_right) = self._split(
            X.copy(), y.copy(), node.col, node.val
        )
        # max depth | min_samples | not delimeters
        if (
            current_depth == self.max_depth
            or len(X) < self.min_samples_split
            or not self._check_delimeters_existence(X)
        ):
            self._add_leaf(node, y_left, "left")
            self._add_leaf(node, y_right, "right")

        # if min_samples_split | _check_delimeters_existence | leafs count
        elif (self.leafs_cnt + self.required_leafs) == self.max_leafs:
            if node.left is None:
                self._add_leaf(node, y_left, "left")
            # right
            if node.right is None:
                self._add_leaf(node, y_right, "right")

        # build new node
        else:
            # find best splits
            left_best_split = self.get_best_split(X_left, y_left)
            right_best_split = self.get_best_split(X_right, y_right)

            # add left split
            if left_best_split[2] == 0 or len(X_left) < self.min_samples_split:
                self._add_leaf(node, y_left, "left")
            else:
                node.left = Node(
                    left_best_split[0],
                    left_best_split[1],
                    self._calculate_criterion(y_left),
                    len(y_left),
                    None,
                    None,
                )
                self.required_leafs += 1
                self._dfs(X_left, y_left, node.left, current_depth + 1)

            # if min_samples_split | _check_delimeters_existence | leafs count
            if (self.leafs_cnt + self.required_leafs) == self.max_leafs:
                if node.left is None:
                    self._add_leaf(node, y_left, "left")
                # right
                if node.right is None:
                    self._add_leaf(node, y_right, "right")
                return None

            # add right split
            if right_best_split[2] == 0 or len(X_right) < self.min_samples_split:
                self._add_leaf(node, y_right, "right")
                return None

            node.right = Node(
                right_best_split[0],
                right_best_split[1],
                self._calculate_criterion(y_right),
                len(y_right),
                None,
                None,
            )
            self.required_leafs += 1
            self._dfs(X_right, y_right, node.right, current_depth + 1)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # find delimeters
        for col_name in X.columns:
            vals = X[col_name].values
            sorted_indices = np.argsort(vals)
            vals = vals[sorted_indices]
            # get delimeters
            _, delimeters = np.unique(vals, return_index=True)
            if self.bins is None or len(delimeters) <= self.bins - 1:
                delimeters = [
                    (vals[delimeter] + vals[delimeter + 1]) / 2
                    for delimeter in delimeters[:-1]
                ]
            else:
                _, delimeters = np.histogram(vals, bins=self.bins)
                delimeters = delimeters[1:-1]

            self.delimeters[col_name] = delimeters

        # fillin fi
        for col_name in X.columns:
            self.fi[col_name] = 0

        # get first split
        col, val, _ = self.get_best_split(X.copy(), y.copy())  # (col, val, ig)
        criterion_val = self._calculate_criterion(y.values)
        self.tree = Node(col, val, criterion_val, len(X), None, None)

        # go deeper
        self.required_leafs = 2
        self._dfs(
            X,
            y,
            self.tree,
            1,
        )

        # update fi
        self._update_fi(self.tree)

    def _update_fi(self, node):
        if isinstance(node, Node):
            left = node.left
            right = node.right
            local_fi = (
                node.elements
                / self.examples_num
                * (
                    node.criterion_val
                    - left.elements / node.elements * left.criterion_val
                    - right.elements / node.elements * right.criterion_val
                )
            )
            self.fi[node.col] += local_fi
            self._update_fi(node.left)
            self._update_fi(node.right)

    def _tree_traversal(self, d: pd.DataFrame):
        node = self.tree
        while not isinstance(node, Leaf):
            if d[node.col] > node.val:
                node = node.right
            else:
                node = node.left
        return node.pred

    def predict(self, X: pd.DataFrame):
        preds = []
        for _, d in X.iterrows():
            pred = self._tree_traversal(d.copy())
            preds.append(pred)
        return preds

    def print_tree(self, node, depth=0):
        if depth == 0:
            print(f"leafs_cnt: {self.leafs_cnt}")
        if isinstance(node, Leaf):
            print("  " * depth, end="")
            print(node.name, "-", node.pred)
        elif isinstance(node, Node):
            print("  " * depth, end="")
            print(node.col, ">", node.val)
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


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


class MyBaggingReg:
    def __init__(
        self,
        estimator=None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: int = 42,
        oob_score: Optional[str] = None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        self.oob_score_ = 0

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _get_oob_score(self, X, y, oob_preds: dict):
        # average preds
        oob_preds = {k: np.mean(v) for k, v in oob_preds.items()}
        y = y[list(oob_preds.keys())]
        y_hat = [oob_preds.get(i, 0) for i in y.index]
        if self.oob_score is None:
            return None
        elif self.oob_score == "mae":
            metric_val = np.mean(np.abs(y - y_hat))
        elif self.oob_score == "mse":
            metric_val = np.mean((y - y_hat) ** 2)
        elif self.oob_score == "rmse":
            metric_val = np.mean((y - y_hat) ** 2)
            metric_val = np.sqrt(metric_val)
        elif self.oob_score == "mape":
            metric_val = 100 * np.mean(np.abs((y - y_hat) / y))
        elif self.oob_score == "r2":
            metric_val = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

        self.oob_score_ = metric_val

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        random.seed(self.random_state)
        self.estimators = []
        # samples
        rows_samples = []
        oob_preds = defaultdict(list)
        for _ in range(self.n_estimators):
            sample_rows_idx = random.choices(
                list(X.index), k=int(len(X) * self.max_samples)
            )
            rows_samples.append(sample_rows_idx)

        for idx in range(self.n_estimators):
            # get sample

            X_train, y_train = X.iloc[rows_samples[idx]], y.iloc[rows_samples[idx]]
            X_test = X.loc[~X.index.isin(rows_samples[idx])].copy()

            # train model
            model = deepcopy(self.estimator)
            model.fit(X_train, y_train)
            self.estimators.append(model)

            # oob preds
            preds = model.predict(X_test)
            for idx, pred in zip(X_test.index, preds):
                oob_preds[idx].append(pred)
        # get oob_score_
        self._get_oob_score(X, y, oob_preds)

    def predict(self, X: pd.DataFrame):
        preds = []
        for model in self.estimators:
            model_pred = model.predict(X.copy())
            preds.append(model_pred)

        # get average
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
