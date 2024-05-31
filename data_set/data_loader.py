import gzip
import pickle
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_regression

from common import BinModel


class DataSetModel(BinModel):
    trainX: Optional[np.ndarray]
    trainY: Optional[np.ndarray]
    validX: Optional[np.ndarray]
    validY: Optional[np.ndarray]
    testX: Optional[np.ndarray]
    testY: Optional[np.ndarray]


class DataLoader(object):

    @classmethod
    def load_mnist(cls):
        with gzip.open("data_set/mnist.pkl.gz", "rb") as fb:
            ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(fb, encoding='latin-1')
            print(x_train, type(x_train))
            return DataSetModel(
                trainX=x_train,
                trainY=y_train,
                validX=x_valid,
                validY=y_valid,
                testX=x_test,
                testY=y_test
            )

    @classmethod
    def randomize_classifier_data(cls):
        """
        Number of informative, redundant and repeated features must sum to less than the number of total features

        n_classes(2) * n_clusters_per_class(2) must be smaller or equal 2**n_informative(1)=2
        """
        X, y = make_classification(
            n_samples=1000, n_features=20,
            n_informative=15, n_redundant=5,
            random_state=7
        )
        return DataSetModel(trainX=X, trainY=y)

    @classmethod
    def randomize_regression_data(cls):
        X, y = make_regression(
            n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7
        )
        return DataSetModel(trainX=X, trainY=y)


data_loader = DataLoader()

if __name__ == '__main__':
    """
    python -m data_set.data_loader
    """
    mnist = data_loader.load_mnist()
    random_classifier_data = data_loader.randomize_classifier_data()
    # plot make classifier
    red_points = random_classifier_data.trainX[random_classifier_data.trainY == 1]
    blue_points = random_classifier_data.trainX[random_classifier_data.trainY == 0]
    plt.scatter(red_points[:, 0], red_points[:, 1], color="red")
    plt.scatter(blue_points[:, 0], blue_points[:, 1], color="blue")
    plt.show()

