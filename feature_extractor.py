import numpy as np
from sklearn.decomposition import PCA


def _compute_mean_signal(sample_signal_array):
    print("computing mean signal")
    return np.mean(sample_signal_array, axis=1).reshape(-1,1)


class FeatureExtractor():
    """
    This class needs an array of shape (samples, signal) to extract the features.

    Features available :
    PCA - Dimension reduction, creates a new signal
    """

    def __init__(self, training, n_PCA = 10):
        """

        """
        self.training = training
        self.features_array = None
        self.pca = PCA(n_components=n_PCA)


    def extract_features(self, sample_signal_array):
        self.feature_array = self._perform_PCA(sample_signal_array)
        self.feature_array = np.concatenate((self.feature_array, _compute_mean_signal(sample_signal_array)), axis=1)
        self.training = False # At the end, training set to false so we do not train PCA or other on test set
        return self.feature_array


    def _perform_PCA(self, sample_signal_array):
        print("performing PCA")
        if self.training:
            self.pca = self.pca.fit_transform(sample_signal_array)
            return self.pca
        else:
            return self.pca.transform(sample_signal_array)



