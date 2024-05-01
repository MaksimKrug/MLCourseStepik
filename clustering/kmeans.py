import numpy as np
import pandas as pd
from typing import Optional
from copy import deepcopy


class MyKMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 10,
        n_init: int = 3,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _euclidean_dist(self, row: np.ndarray, centroids: np.ndarray):
        distances = np.sqrt(np.sum((row - centroids) ** 2, axis=1))
        return np.argmin(distances)

    def _calculate_wcss(self, X: pd.DataFrame, centroids: list):
        # find clusters
        clusters = np.apply_along_axis(
            self._euclidean_dist, 1, X.values, centroids=centroids
        )
        # find wcss
        res = 0
        for cluster_idx in range(self.n_clusters):
            cluster_points = X.values[np.where(clusters == cluster_idx)]
            if cluster_points.size != 0:
                euclidean_dist = np.sum(
                    (cluster_points - centroids[cluster_idx]) ** 2, axis=1
                )
                res += np.sum(euclidean_dist)
        return res

    def fit(self, X: pd.DataFrame):
        # utils
        np.random.seed(self.random_state)
        all_centroids = []

        for _ in range(self.n_init):
            # find init centroids
            centroids = [
                [np.random.uniform(X[col].min(), X[col].max()) for col in X.columns]
                for _ in range(self.n_clusters)
            ]
            centroids = np.asarray(centroids)
            # fit
            for iter_num in range(1, self.max_iter + 1):
                clusters = np.apply_along_axis(
                    self._euclidean_dist, 1, X.values, centroids=centroids
                )
                # update centroids
                new_centroids = deepcopy(centroids)
                for cluster_idx in range(self.n_clusters):
                    cluster_points = X.values[np.where(clusters == cluster_idx)]
                    if cluster_points.size == 0:
                        continue
                    # update centroids
                    new_centroids[cluster_idx] = cluster_points.mean(0)

                # break statement
                if np.allclose(new_centroids, centroids):
                    break
                centroids = new_centroids

            # update all_centroids
            all_centroids.append(centroids)

        # find best centroids
        clustering_metrics = [
            self._calculate_wcss(X, centroids) for centroids in all_centroids
        ]
        self.cluster_centers_ = all_centroids[np.argmin(clustering_metrics)]
        self.inertia_ = clustering_metrics[np.argmin(clustering_metrics)]

    def predict(self, X: pd.DataFrame):
        clusters = np.apply_along_axis(
            self._euclidean_dist, 1, X.values, centroids=self.cluster_centers_
        )
        return clusters
