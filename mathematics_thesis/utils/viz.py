import jax.numpy as np

import matplotlib.pyplot as plt


def format_plot(x=None, y=None):
    # plt.grid(False)
    ax = plt.gca()
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()


def plot_fn(train, test, *fs):
    train_xs, train_ys = train

    plt.plot(train_xs, train_ys, 'ro', markersize=4, label='train')

    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

        for f in fs:
            plt.plot(test_xs, f(test_xs), '-', linewidth=3)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])

    format_plot('$x$', '$f$')


def plot_interpolation(train, x_test, *fs):
    x_train, y_train = train
    train_alphas = np.array((0, 1))
    plt.plot(train_alphas, y_train, 'ro', markersize=5, label='train')

    if x_test != None:
        # plt.plot(x_test, test_ys, 'k--', linewidth=3, label='$f(x)$')
        for f in fs:
            plt.plot(x_test, f(x_test), '-', linewidth=3)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])

    format_plot(r'$\alpha$', r'$f$')