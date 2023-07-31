import jax.numpy as np

import matplotlib.pyplot as plt


def format_plot(x=None, y=None):
    # plt.grid(False)
    ax = plt.gca()
    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()


def plot_fn(train, test, xlabel=None, ylabel=None, *fs):
    train_xs, train_ys = train
    plt.plot(train_xs, train_ys, 'mo', markersize=3, label='Train')

    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, 'k--', linewidth=1.5, label='$f(x)$')

        for f in fs:
            plt.plot(test_xs, f(test_xs), '-', linewidth=1.5)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])

    format_plot(xlabel, ylabel)


def plot_interpolation(train, x_test, xlabel=None, ylabel=None, *fs):
    x_train, y_train = train
    train_alphas = np.array((1, 0))
    plt.plot(train_alphas, y_train, 'mo', markersize=3, label='train')

    if x_test != None:
        # plt.plot(x_test, test_ys, 'k--', linewidth=3, label='$f(x)$')
        for f in fs:
            plt.plot(x_test, f(x_test), '-', linewidth=1.5)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])

    format_plot(xlabel, ylabel)