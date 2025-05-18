import typing
from collections import Counter
import numpy as np
# import argparse
# import matplotlib.pyplot as plt

from numpy.typing import NDArray

type ArrayType = NDArray[typing.Any]


def euclidean_dist(x1: ArrayType, x2: ArrayType) -> float:
    """
    Calculate the Euclidean distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Euclidean distance between the two data points.
    """
    return float(np.linalg.norm(x1 - x2))


def manhattan_dist(x1: ArrayType, x2: ArrayType) -> float:
    """
    Calculate the Manhattan distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Manhattan distance between the two data points.
    """
    return np.sum(np.abs(x1 - x2))


def predict(
    k: int,
    dist_metric: typing.Callable[[ArrayType, ArrayType], float],
    train_data: ArrayType,
    X: ArrayType,
):
    """
    Compute the predictions for the data points in X using a kNN
    classified defined by the first three parameters

    Parameters:
        k (int): Number of nearest neighbors to use when computing
                the predictions
        dist_metric (int): Distance metric to use when computing
                            the predictions
        train_data (np.ndarray): Training dataset
        X (np.ndarray): Feature values of the data points to be
                        predicted

    Returns:
        np.ndarray: Vector of predicted labels
    """
    nearest: list[tuple[float, int]] = []
    for vector in train_data:
        distance = dist_metric(X, vector[:-1])
        nearest.append((distance, int(vector[-1])))
    nearest.sort(key=lambda t: t[0])
    return majority(k, nearest)


def majority(k: int, nearest: list[tuple[float, int]]) -> int:
    nearest = nearest[:k]
    labels = [t[1] for t in nearest]
    return most_frequent_elements(labels)


def most_frequent_elements(labels: list[int]) -> int:
    count = Counter(labels)
    max_freq = max(count.values())
    candidates = set([elem for elem, freq in count.items() if freq == max_freq])
    return min(candidates)


def compute_error(preds: ArrayType, labels: ArrayType) -> float:
    """
    Compute the error rate for a given set of predictions
    and labels

    Parameters:
        preds (np.ndarray): Your models predictions
        labels (np.ndarray): The correct labels

    Returns:
        float: Error rate of the predictions
    """
    assert preds.size == labels.size
    return np.sum(preds != labels) / preds.size


def val_model(
    ks: range,
    dist_metric: typing.Callable[[ArrayType, ArrayType], float],
    train_data: ArrayType,
    val_data: ArrayType,
):
    """
    For each value in ks, compute the training and validation
    error rates

    Parameters:
        ks (range): Set of values
        dist_metric (int): Distance metric to use when computing
                            the predictions
        train_data (np.ndarray): Training dataset
        val_data (np.ndarray): Validation dataset

    Returns:
        tuple(train_preds: ArrayType,
              val_preds: ArrayType,
              train_errs: ArrayType,
              val_errs: ArrayType): tuple of predictions and error rate arrays
                                    where each row corresponds to one of the k
                                    values in ks. For the preds arrays, the length
                                    of each row will be the number of data points
                                    in the relevant dataset and for the errs arrays,
                                     each row will consistent of a single error rate.
    """

    train_preds: list[ArrayType] = []
    val_preds: list[ArrayType] = []
    train_errs: list[float] = []
    val_errs: list[float] = []

    for k in ks:
        print(f"\n{k=}")

        def curried_predict(X: ArrayType):
            return predict(k, dist_metric, train_data, X)

        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        train_pred = np.apply_along_axis(curried_predict, 1, train_features)
        train_err = compute_error(train_pred, train_labels)
        print(f"{train_err=}")

        validate_features, validate_labels = val_data[:, :-1], val_data[:, -1]
        val_pred = np.apply_along_axis(curried_predict, 1, validate_features)
        val_err = compute_error(val_pred, validate_labels)
        print(f"{val_err=}")

        train_preds.append(train_pred)
        val_preds.append(val_pred)
        train_errs.append(train_err)
        val_errs.append(val_err)

    return (train_preds, val_preds, train_errs, val_errs)


def crossval_model(
    ks: range,
    num_folds: int,
    dist_metric: typing.Callable[[ArrayType, ArrayType], float],
    train_data: ArrayType,
):
    """
    For each value in ks, compute the cross-validation error rate

    Parameters:
        ks (range): Set of values
        dist_metric (int): Distance metric to use when computing
                                           the predictions
        num_folds(int): number of folds to split the training
                                        dataset into
        train_data (np.ndarray): Training dataset

    Returns:
        tuple(crossval_preds: ArrayType,
              crossval_errs: ArrayType): tuple of predictions and error rate arrays
                                          where each row corresponds to one of the k
                                          values in ks. For the preds array, the
                                          length of each row will be the number of
                                          training data points and should contain the
                                          prediction for the corresponding data point
                                          when held out as a validation data point.
                                          For the errs array, each row will contain the
                                          corresponding num_folds-fold cross-validation
                                          error rate.
    """
    # TODO: Implement cross-validation
    pass


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("train_input", type=str, help="path to formatted training data")
    # parser.add_argument("val_input", type=str, help="path to formatted validation data")
    # parser.add_argument("test_input", type=str, help="path to formatted test data")
    # parser.add_argument(
    #     "val_type",
    #     type=int,
    #     choices=[0, 1],
    #     help="validation type; 0 = validation, 1 = cross-validation",
    # )
    # parser.add_argument(
    #     "dist_metric",
    #     type=int,
    #     choices=[0, 1],
    #     help="distance metric; 0 = Euclidean, 1 = Manhattan",
    # )
    # parser.add_argument(
    #     "min_k", type=int, help="smallest value of k to consider (inclusive)"
    # )
    # parser.add_argument(
    #     "max_k", type=int, help="largest value of k to consider (inclusive)"
    # )
    # parser.add_argument(
    #     "train_out", type=str, help="file to write train predictions to"
    # )
    # parser.add_argument("val_out", type=str, help="file to write train predictions to")
    # parser.add_argument("test_out", type=str, help="file to write test predictions to")
    # parser.add_argument("metrics_out", type=str, help="file to write metrics to")
    # args = parser.parse_args()

    train_data = np.genfromtxt("iris_train.csv", delimiter=",", skip_header=1)
    val_data = np.genfromtxt("iris_val.csv", delimiter=",", skip_header=1)
    # print(predict(5, euclidean_dist, train_data, np.array([5.8,2.7,3.9,1.2])))
    val_model(range(1, 5), euclidean_dist, train_data, val_data)
