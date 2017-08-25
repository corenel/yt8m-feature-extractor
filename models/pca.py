"""Wrapper for PCA estimator."""

from sklearn.decomposition import PCA


class PCAWrapper(object):
    """Wrapper for PCA estimator."""

    def __init__(self, n_components):
        """Init PCA."""
        super(PCAWrapper, self).__init__()
        self.n_components = n_components
        self.estimator = PCA(n_components=self.n_components,
                             whiten=True)

    def fit(self):
        """Fit data."""
        pass

    def transform(self, input):
        """Apply dimensionality reduction to input."""
        pass

    def save_params(self, filepath):
        """Save params of PCA."""
        pass

    def load_params(self, filepath):
        """Load params of PCA."""
        pass
