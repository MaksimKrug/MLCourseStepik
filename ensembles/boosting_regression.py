import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Union, Optional, Callable
from collections import defaultdict

import random


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


class MyBoostReg:
    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: Union[Callable[[float], float], float] = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = 16,
        loss: str = "MSE",
        metric: Optional[str] = None,
        max_features: float = 0.5,
        max_samples: float = 0.5,
        random_state: int = 42,
        reg: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.metric = metric

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = 0
        self.best_score = 0 if self.metric == "R2" else np.inf
        self.trees = []
        self.fi = {}

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _update_leafs(self, node: Union[Node, Leaf], X, y, preds):
        if isinstance(node, Leaf):
            residual = y - preds
            if self.loss == "MSE":
                node.pred = np.mean(residual) + self.leafs_cnt * self.reg
            elif self.loss == "MAE":
                node.pred = np.median(residual) + self.leafs_cnt * self.reg
        elif isinstance(node, Node):
            # left
            left_idx = X[node.col] <= node.val
            X_left, y_left, preds_left = X[left_idx], y[left_idx], preds[left_idx]
            self._update_leafs(node.left, X_left, y_left, preds_left)
            # right
            right_idx = X[node.col] > node.val
            X_right, y_right, preds_right = X[right_idx], y[right_idx], preds[right_idx]
            self._update_leafs(node.right, X_right, y_right, preds_right)

    def _calculate_scores(self, y_true, y_pred):
        # loss
        loss = self._calculate_loss(y_true, y_pred)
        # metric
        if self.metric is None:
            metric_val = loss
        elif self.metric == "MAE":
            metric_val = np.mean(np.abs(y_true - y_pred))
        elif self.metric == "MSE":
            metric_val = np.mean((y_pred - y_true) ** 2)
        elif self.metric == "RMSE":
            metric_val = np.mean((y_true - y_pred) ** 2)
            metric_val = np.sqrt(metric_val)
        elif self.metric == "MAPE":
            metric_val = 100 * np.mean(np.abs((y_true - y_pred) / y))
        elif self.metric == "R2":
            metric_val = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
                (y_true - y_true.mean()) ** 2
            )

        return loss, metric_val

    def _calculate_loss(self, y_true, y_pred):
        if self.loss == "MSE":
            loss = np.mean((y_pred - y_true) ** 2)
        elif self.loss == "MAE":
            loss = np.mean(np.abs(y_pred - y_true))
        return loss

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
        early_stopping: Optional[int] = None,
        verbose: Optional[int] = None,
    ):
        random.seed(self.random_state)

        self.leafs_cnt = 0
        no_improvements = 0
        if self.loss == "MSE":
            self.pred_0 = np.mean(y)
        elif self.loss == "MAE":
            self.pred_0 = np.median(y)
        preds = pd.Series(np.full_like(y, self.pred_0), index=y.index)
        if early_stopping:
            pred_eval_0 = np.mean(y_eval) if self.loss == "MSE" else np.median(y_eval)
            preds_eval = pd.Series(
                np.full_like(y_eval, pred_eval_0), index=y_eval.index
            )

        for iter_num in range(1, self.n_estimators + 1):
            # sample
            cols_idx = random.sample(
                list(X.columns), round(X.shape[1] * self.max_features)
            )
            rows_idx = random.sample(
                range(X.shape[0]), round(X.shape[0] * self.max_samples)
            )
            X_sample = X.loc[rows_idx, cols_idx].copy()
            y_sample = y.loc[rows_idx].copy()
            preds_sample = preds.loc[rows_idx].copy()

            # gradient
            if self.loss == "MSE":
                grad = 2 * (preds - y)
            elif self.loss == "MAE":
                grad = np.sign(preds - y)

            loss, score = self._calculate_scores(y, preds)
            print("AAA", loss, score)

            grad = grad.loc[rows_idx].copy()

            # get sample
            model = MyTreeReg(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
                examples_num=len(y),
            )

            # fit
            model.fit(X_sample, -grad)
            # update leafs
            self._update_leafs(model.tree, X_sample, y_sample, preds_sample)
            self.trees.append(model)
            # update preds
            new_preds = model.predict(X)
            if callable(self.learning_rate):
                lr_step = self.learning_rate(iter_num)
            else:
                lr_step = self.learning_rate
            preds += np.asarray(new_preds) * lr_step
            self.leafs_cnt += model.leafs_cnt

            # loss & metrics
            if early_stopping:
                # get scores
                model_eval_preds = model.predict(X_eval)
                preds_eval += np.asarray(model_eval_preds) * lr_step
                loss, score = self._calculate_scores(y_eval, preds_eval)
                if self.metric == "R2":
                    no_improvements += score < self.best_score
                    self.best_score = max(self.best_score, score)
                else:
                    no_improvements += score > self.best_score
                    self.best_score = min(self.best_score, score)

                if no_improvements == early_stopping:
                    self.trees = self.trees[:-early_stopping]
                    break
            else:
                preds_sample = preds.loc[rows_idx].copy()
                loss, score = self._calculate_scores(y_sample, preds_sample)
                self.best_score = score

            # verbose
            if verbose is not None and iter_num % verbose == 0:
                print(f"{iter_num}. Loss[{self.loss}]: {loss} | {self.metric}: {score}")

        # update fi
        for col in X.columns:
            self.fi[col] = 0.0
        for tree in self.trees:
            for col in X.columns:
                self.fi[col] += tree.fi.get(col, 0)

    def predict(self, X: pd.DataFrame):
        preds = []
        for iter_num, tree in enumerate(self.trees, start=1):
            pred = tree.predict(X)
            pred = np.asarray(pred)
            if callable(self.learning_rate):
                lr_step = self.learning_rate(iter_num)
            else:
                lr_step = self.learning_rate
            preds.append(pred * lr_step)

        preds = np.asarray(preds).sum(axis=0)
        preds += self.pred_0

        return preds


from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1500, n_features=14, n_informative=10, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f"col_{col}" for col in X.columns]
X_train, X_eval = X.iloc[:1200], X.iloc[1200:]
y_train, y_eval = y.iloc[:1200], y.iloc[1200:]

obj = MyBoostReg(
    loss="MSE", metric="RMSE", max_features=0.3, max_samples=0.3, learning_rate=1.6
)
obj.fit(X_train, y_train, X_eval, y_eval, 3, verbose=1)
