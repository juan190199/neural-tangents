from jax import random

import jax.numpy as np


def loss_fn(predict_fn, ys, t, xs=None, get='ntk'):
    mean, cov = predict_fn(t=t, get=get, x_test=xs, compute_cov=True)
    mean = np.reshape(mean, mean.shape[:1] + (-1,))
    var = np.diagonal(cov, axis1=1, axis2=2)
    ys = np.reshape(ys, (1, -1))

    mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2, axis=1)
    var_predictions = 0.5 * np.var(ys ** 2 - 2 * mean * ys + var + mean ** 2, axis=1)

    return mean_predictions, var_predictions


def interpolate_points(x1, x2, alpha):
    return alpha * x1 + (1 - alpha) * x2


# def choose_random_idxs(key, labels, n_idxs=2):
#     unique_labels = np.unique(labels)
#     label1, label2 = random.choice(key, unique_labels, shape=(2,), replace=False)
#     idxs_label1 = np.where(labels == label1)[0]
#     idxs_label2 = np.where(labels == label2)[0]
#
#     key, subkey1, subkey2 = random.split(key, 3)
#     idx1 = random.choice(subkey1, idxs_label1)
#     idx2 = random.choice(subkey2, idxs_label2)
#
#     return idx1, idx2


def choose_random_idxs(key, labels, n_idxs=2):
    unique_labels = np.unique(labels)
    n_labels = unique_labels.shape[0]

    if n_labels < n_idxs:
        raise ValueError('Number of indices to select is greater than number of unique labels.')

    # ToDo: Raise error when there are not enough labels to pick at least n_idx data point with different labels

    idxs = np.array([], dtype=np.int32)
    for _ in range(n_idxs):
        key, subkey = random.split(key)
        label = random.choice(subkey, unique_labels)
        label_idxs = np.where(labels == label)[0]

        key, subkey = random.split(key)
        idx = random.choice(subkey, label_idxs)
        idxs = np.append(idxs, idx)
        labels = np.delete(labels, idx)

    return idxs.astype(int)


    label1, label2 = random.choice(key, unique_labels, shape=(2,), replace=False)
    idxs_label1 = np.where(labels == label1)[0]
    idxs_label2 = np.where(labels == label2)[0]

    key, subkey1, subkey2 = random.split(key, 3)
    idx1 = random.choice(subkey1, idxs_label1)
    idx2 = random.choice(subkey2, idxs_label2)

    return idx1, idx2


def accuracy_score(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred")

    # Flatten the arrays if they have shape (len(data), 1)
    if len(y_true.shape) == 2:
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    return accuracy
