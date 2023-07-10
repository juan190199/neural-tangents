from jax.example_libraries import optimizers
import jax.numpy as np


@optimizers.optimizer
def momentum(learning_rate, momentum=0.9):
    """A standard momentum optimizer for testing.

    Different from `jax.example_libraries.optimizers.momentum` (Nesterov).
    """
    learning_rate = optimizers.make_schedule(learning_rate)

    def init_fn(x0):
        v0 = np.zeros_like(x0)
        return x0, v0

    def update_fn(i, g, state):
        x, velocity = state
        velocity = momentum * velocity + g
        x = x - learning_rate(i) * velocity
        return x, velocity

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update_fn, get_params