import typing
from collections import Counter
import numpy as np
import math
import argparse
# import matplotlib.pyplot as plt

from numpy.typing import NDArray

type ArrayType = NDArray[typing.Any]

FOLD_NUMBER = 10


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


def predict_all(
    k: int,
    dist_metric: typing.Callable[[ArrayType, ArrayType], float],
    train_data: ArrayType,
    Xs: ArrayType,
):
    def curried_predict(X: ArrayType):
        return predict(k, dist_metric, train_data, X)

    return np.apply_along_axis(curried_predict, 1, Xs)


def predict(
    k: int,
    dist_metric: typing.Callable[[ArrayType, ArrayType], float],
    train_data: ArrayType,
    X: ArrayType,
) -> int:
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

    # print(f"{train_data.shape=}, {val_data.shape=}")

    train_preds: list[ArrayType] = []
    val_preds: list[ArrayType] = []
    train_errs: list[float] = []
    val_errs: list[float] = []

    for k in ks:
        # print(f"\n{k=}")

        def curried_predict(X: ArrayType):
            return predict(k, dist_metric, train_data, X)

        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        train_pred = np.apply_along_axis(curried_predict, 1, train_features)
        train_err = compute_error(train_pred, train_labels)
        # print(f"{train_err=}")

        validate_features, validate_labels = val_data[:, :-1], val_data[:, -1]
        val_pred = np.apply_along_axis(curried_predict, 1, validate_features)
        val_err = compute_error(val_pred, validate_labels)
        # print(f"{val_err=}")

        train_preds.append(train_pred)
        val_preds.append(val_pred)
        train_errs.append(train_err)
        val_errs.append(val_err)

    return (train_preds, val_preds, train_errs, np.array(val_errs))


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

    # print(f"{train_data.shape=}")

    step = math.ceil(len(train_data) / num_folds)

    crossval_preds: list[list[ArrayType]] = []
    crossval_errs: list[ArrayType] = []

    for i in range(num_folds):
        # print(f"\n======================== Fold {i} ========================")
        val_data = train_data[i * step : (i + 1) * step, :]
        _train_data = np.concatenate(
            (train_data[: i * step, :], train_data[(i + 1) * step :, :])
        )
        (_, val_preds, _, val_errs) = val_model(ks, dist_metric, _train_data, val_data)
        # val_preds = (k, n / fold)
        # val_errs  = (k,  )
        crossval_preds.append(val_preds)
        crossval_errs.append(val_errs)
    preds_result = np.concatenate(tuple(crossval_preds), axis=1)
    errors_result = np.mean(np.array(crossval_errs), axis=0)
    # print(f"{preds_result.shape=}")
    # print(f"{errors_result.shape=}, \n{errors_result=}")
    return (preds_result, errors_result)


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help="path to formatted training data")
    parser.add_argument("val_input", type=str, help="path to formatted validation data")
    parser.add_argument("test_input", type=str, help="path to formatted test data")
    parser.add_argument(
        "val_type",
        type=int,
        choices=[0, 1],
        help="validation type; 0 = validation, 1 = cross-validation",
    )
    parser.add_argument(
        "dist_metric",
        type=int,
        choices=[0, 1],
        help="distance metric; 0 = Euclidean, 1 = Manhattan",
    )
    parser.add_argument(
        "min_k", type=int, help="smallest value of k to consider (inclusive)"
    )
    parser.add_argument(
        "max_k", type=int, help="largest value of k to consider (inclusive)"
    )
    parser.add_argument(
        "train_out", type=str, help="file to write train predictions to"
    )
    parser.add_argument("val_out", type=str, help="file to write train predictions to")
    parser.add_argument("test_out", type=str, help="file to write test predictions to")
    parser.add_argument("metrics_out", type=str, help="file to write metrics to")
    args = parser.parse_args()

    # train_data = np.genfromtxt("iris_train.csv", delimiter=",", skip_header=1)
    # val_data = np.genfromtxt("iris_val.csv", delimiter=",", skip_header=1)
    # print(predict(5, euclidean_dist, train_data, np.array([5.8,2.7,3.9,1.2])))
    # val_model(range(1, 5), euclidean_dist, train_data, val_data)
    # crossval_model(
    #     range(1, 11), 10, manhattan_dist, np.concatenate((train_data, val_data), axis=0)
    # )

    dist_metric = euclidean_dist if args.dist_metric == 0 else manhattan_dist
    train_data = np.genfromtxt(args.train_input, delimiter=",", skip_header=1)
    val_input = np.genfromtxt(args.val_input, delimiter=",", skip_header=1)
    test_input = np.genfromtxt(args.test_input, delimiter=",", skip_header=1)

    match args.val_type:
        case 0:
            (train_preds, val_preds, train_errs, val_errs) = val_model(
                range(args.min_k, args.max_k + 1), dist_metric, train_data, val_input
            )

            with open(args.train_out, "w") as fout:
                for preds in train_preds:
                    fout.write(",".join(map(str, list(preds))))
                    fout.write("\n")

            with open(args.val_out, "w") as fout:
                for preds in val_preds:
                    fout.write(",".join(map(str, list(preds))))
                    fout.write("\n")

            best_k = range(args.min_k, args.max_k + 1)[val_errs.argmin()]
            predict_labels_only_train = predict_all(
                best_k, dist_metric, train_data, test_input[:, :-1]
            )
            predict_labels_using_train_and_val = predict_all(
                best_k, dist_metric, np.concatenate((train_data, val_input)), test_input[:, :-1]
            )

            with open(args.metrics_out, 'w') as fout:
                for idx, err in enumerate(train_errs):
                    fout.write(f"k={idx+1} training error rate: {err}\n")

                for idx, err in enumerate(val_errs):
                    fout.write(f"k={idx+1} validation error rate: {err}\n")

                fout.write(f"test error rate (train): {compute_error(predict_labels_only_train, test_input[:, -1])}\n")
                fout.write(f"test error rate (train + validation): {compute_error(predict_labels_using_train_and_val, test_input[:, -1])}\n")

            with open(args.test_out, "w") as fout:
                for p in predict_labels_using_train_and_val:
                    fout.write(str(p))
                    fout.write("\n")
        case _:
            train_data = np.concatenate((train_data, val_input))
            (preds_result, errors_result) = crossval_model(
                range(args.min_k, args.max_k + 1),
                FOLD_NUMBER,
                dist_metric,
                train_data,
            )
            best_k: int = range(args.min_k, args.max_k + 1)[errors_result.argmin()] #type: ignore
            predict_labels = predict_all(
                best_k, dist_metric, train_data, test_input[:, :-1] #type: ignore
            )
            with open(args.val_out, 'w') as fout:
                for preds in preds_result:
                    fout.write(",".join(map(str, list(preds))))
                    fout.write("\n")

            with open(args.test_out, "w") as fout:
                for p in predict_labels:
                    fout.write(str(p))
                    fout.write("\n")

            with open(args.metrics_out, 'w') as fout:
                for idx, error in enumerate(errors_result):
                    fout.write(f"k={idx+1} cross-validation error rate: {error}\n")
                fout.write(f"test error rate: {compute_error(predict_labels, test_input[:, -1])}\n")
                