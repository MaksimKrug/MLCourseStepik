import numpy as np
import pandas as pd


class MyPCA:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def fit_transform(self, X: pd.DataFrame):
        # data = X.values
        mean = X.mean(0)
        X = X - mean
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        main_components = eigenvectors[:, -self.n_components :]
        res = X @ main_components
        return pd.DataFrame(res)
