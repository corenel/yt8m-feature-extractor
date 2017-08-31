"""Wrapper for PCA estimator."""

import os

from sklearn.decomposition import PCA
from sklearn.externals import joblib

# import pickle


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
        abs_path = os.path.abspath(filepath)
        if not os.path.exists(os.path.dirname(abs_path)):
            os.makedirs(os.path.dirname(abs_path))
        # pickle.dump(self.estimator, open(abs_path, "wb"))
        joblib.dump(self.estimator, abs_path)
        print("save PCA params to {}".format(abs_path))

    def load_params(self, filepath):
        """Load params of PCA."""
        if os.path.exists(filepath):
            # self.estimator = pickle.load(open(filepath, "rb"))
            self.estimator = joblib.load(filepath)
            print("load PCA params from {}".format(filepath))
        else:
            print("PCA params doesn't exist in {}".format(filepath))

    def show_params(self):
        """Show params of PCA."""
        print("mean:")
        print(self.estimator.mean_)
        print("explained_variance_ratio:")
        print(self.estimator.explained_variance_ratio_)
        print("singular_values:")
        print(self.estimator.singular_values_)
