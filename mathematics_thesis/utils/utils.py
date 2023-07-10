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


def choose_random_idxs(key, dataset, num_points=2):
    num_samples = dataset.shape[0]
    idxs = random.choice(key, num_samples, shape=(num_points, ), replace=False)
    return idxs
