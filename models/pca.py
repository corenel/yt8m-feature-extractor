"""Wrapper for PCA estimator."""

import os

import numpy as np
from sklearn.decomposition import PCA


class PCAWrapper(object):
    """Wrapper for PCA estimator."""

    def __init__(self, n_components):
        """Init PCA."""
        super(PCAWrapper, self).__init__()
        self.n_components = n_components
        self.estimator = PCA(n_components=self.n_components,
                             whiten=True)

    def fit(self, X):
        """Fit data with X."""
        self.estimator.fit(X)

    def transform(self, X):
        """Apply dimensionality reduction to X."""
        return self.estimator.transform(X)

    def save_params(self, filepath):
        """Save params of PCA."""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        params = self.estimator.save_params()
        np.savez_compressed(filepath, params)

    def load_params(self, filepath):
        """Load params of PCA."""
        if os.path.exists(filepath):
            params = np.load(filepath)
            self.estimator.load_params(params)
        else:
            print("params doesn't exist: {}".format(filepath))
