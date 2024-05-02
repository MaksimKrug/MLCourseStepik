import numpy as np
import pandas as pd
from collections import Counter


class MyAgglomerative:
    def __init__(self, n_clusters: int = 3, metric: str = "euclidean"):
        self.n_clusters = n_clusters
        self.metric = metric

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _pred(self, row: np.ndarray, centroids: np.ndarray):
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((row - centroids) ** 2, axis=-1))
        elif self.metric == "chebyshev":
            distances = np.abs(row - centroids)
            distances = np.max(distances, axis=-1)
        elif self.metric == "manhattan":
            distances = np.abs(row - centroids).sum(-1)
        elif self.metric == "cosine":
            dot_products = np.dot(centroids, row)
            target_norm = np.linalg.norm(row)
            matrix_norms = np.linalg.norm(centroids, axis=1)
            cosine_similarities = dot_products / (target_norm * matrix_norms)
            distances = 1 - cosine_similarities

        return np.argmin(distances)

    def _get_distances(self, centroids):
        if self.metric == "euclidean":
            distances = np.sqrt(np.square(centroids[:, np.newaxis] - centroids).sum(2))
        elif self.metric == "chebyshev":
            distances = np.abs(centroids[:, np.newaxis] - centroids)
            distances = np.max(distances, axis=2)
        elif self.metric == "manhattan":
            distances = np.abs(centroids[:, np.newaxis] - centroids).sum(2)
        elif self.metric == "cosine":
            dot_products = np.dot(centroids, centroids.T)
            norms = np.linalg.norm(centroids, axis=1)
            cosine_similarities = dot_products / (
                norms[:, np.newaxis] * norms[np.newaxis, :]
            )
            distances = 1 - cosine_similarities

        return distances

    def fit_predict(self, X: pd.DataFrame):
        # initial centroids
        points = X.values
        centroids = X.values
        cluster2points = {i: [i] for i in range(len(points))}
        while len(centroids) != self.n_clusters:
            # find distances
            distances = self._get_distances(centroids)
            np.fill_diagonal(distances, np.inf)
            # min index
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            # find new point
            indexes = cluster2points[min_index[0]] + cluster2points[min_index[1]]
            new_cluster_points = points[[indexes]]
            new_cluster_points = (
                [new_cluster_points]
                if new_cluster_points.ndim == 2
                else new_cluster_points
            )
            new_point = np.mean(new_cluster_points, axis=1)
            # update centroids
            centroids = np.vstack((np.delete(centroids, min_index, axis=0), new_point))

            # update cluster2points
            row1 = cluster2points[min_index[0]]
            row2 = cluster2points[min_index[1]]
            del cluster2points[min_index[0]]
            del cluster2points[min_index[1]]
            cluster2points = {
                idx - (idx > min_index[0]) - (idx > min_index[1]): vals
                for idx, vals in cluster2points.items()
            }
            cluster2points[len(centroids) - 1] = row1 + row2

        # predict
        preds = [
            cluster_idx for cluster_idx, vals in cluster2points.items() for _ in vals
        ]
        return preds
