import numpy as np
import pandas as pd

class MyAgglomerative:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        ) 
    
    def _euclidean_dist(self, row: np.ndarray, centroids: np.ndarray):
        distances = np.sqrt(np.sum((row - centroids) ** 2, axis=1))
        return np.argmin(distances)
    
    def fit_predict(self, X: pd.DataFrame):
        # initial centroids
        points = X.values
        centroids = X.values
        idx2points = {i:[i] for i in range(len(points))}
        while len(centroids) != self.n_clusters:
            # find distances
            distances = np.sqrt(np.square(centroids[:, np.newaxis] - centroids).sum(2))
            np.fill_diagonal(distances, np.inf)
            # min index
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            # find new point
            indexes = idx2points[min_index[0]] + idx2points[min_index[1]]
            new_point = np.mean(centroids[[indexes]], axis=1)
            # update centroids
            centroids = np.vstack((np.delete(centroids, min_index, axis=0), new_point))

            # update idx2points
            print(min_index, idx2points)
            row1 = idx2points[min_index[0]]
            row2 = idx2points[min_index[1]]
            del idx2points[min_index[0]]
            del idx2points[min_index[1]]
            print(idx2points)
            idx2points = {idx-1 if idx > min_index[0] else idx:vals for idx, vals in idx2points.items()}
            print(idx2points)
            idx2points = {idx-1 if idx > min_index[1] else idx:vals for idx, vals in idx2points.items()}
            print(idx2points)
            idx2points[len(centroids)-1] = row1 + row2
            print(idx2points)


        # predict
        preds = np.apply_along_axis(
            self._euclidean_dist, 1, X.values, centroids=centroids
        )
        return preds


from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=10, centers=3, n_features=3, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

obj = MyAgglomerative()
obj.fit_predict(X)