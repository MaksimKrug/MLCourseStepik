import numpy as np
import pandas as pd


class MyDBSCAN:
    def __init__(self, eps: int = 3, min_samples: int = 3, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _get_distance(self, point: np.ndarray, points: np.ndarray):
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((point - points) ** 2, axis=-1))
        elif self.metric == "chebyshev":
            distances = np.abs(point - points)
            distances = np.max(distances, axis=-1)
        elif self.metric == "manhattan":
            distances = np.abs(point - points).sum(-1)
        elif self.metric == "cosine":
            dot_products = np.dot(points, point)
            target_norm = np.linalg.norm(point)
            matrix_norms = np.linalg.norm(points, axis=1)
            cosine_similarities = dot_products / (target_norm * matrix_norms)
            distances = 1 - cosine_similarities

        return distances

    def fit_predict(self, X: pd.DataFrame):
        cluster2points = {0: set()}
        points_to_visit = set(range(len(X)))
        points = np.asarray(X.values, dtype=np.float32)

        while len(points_to_visit) != 0:
            # get new point
            point_idx = points_to_visit.pop()
            point = points[point_idx]
            # get all neighbours
            distances = self._get_distance(point, points)
            distances[point_idx] = np.inf
            neighbours = set(np.where(distances <= self.eps)[0])

            # outlier
            if len(neighbours) < self.min_samples:
                cluster2points[0].add(point_idx)
                continue

            # it's a root
            cluster_num = len(cluster2points)
            cluster2points[cluster_num] = set([point_idx])
            # store point indxes
            neighbours = {
                n for n in neighbours if n in points_to_visit or n in cluster2points[0]
            }
            while len(neighbours) != 0:
                # get neighbour neighbours
                neighbour_idx = neighbours.pop()
                cluster2points[cluster_num].add(neighbour_idx)
                if neighbour_idx in points_to_visit:
                    points_to_visit.remove(neighbour_idx)

                neighbour = points[neighbour_idx]
                distances = self._get_distance(neighbour, points)
                distances[neighbour_idx] = np.inf
                temp_neighbours = set(np.where(distances <= self.eps)[0])
                # if border
                if neighbour_idx in cluster2points[0]:
                    cluster2points[0].remove(neighbour_idx)
                if len(temp_neighbours) < self.min_samples:
                    continue
                # if root
                else:
                    temp_neighbours = {
                        n
                        for n in temp_neighbours
                        if n in points_to_visit or n in cluster2points[0]
                    }
                    neighbours.update(temp_neighbours)

        # predict
        preds = []
        for point_idx in range(len(X)):
            for cluster_num, cluster_points in cluster2points.items():
                if point_idx in cluster_points:
                    preds.append(cluster_num)
                    break
        return preds
