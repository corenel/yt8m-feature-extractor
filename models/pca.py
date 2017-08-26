"""Wrapper for PCA estimator."""

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib


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
        joblib.dump(self.estimator, filepath)
        print("save PCA params to {}".format(filepath))

    def load_params(self, filepath):
        """Load params of PCA."""
        if os.path.exists(filepath):
            self.estimator = joblib.load(filepath)
            print("load PCA params from {}".format(filepath))
        else:
            print("PCA params doesn't exist in {}".format(filepath))
