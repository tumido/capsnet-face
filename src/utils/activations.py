"""Activation functions used by CapsNet."""

import keras.backend as k


def squash(input, axis=-1):
    """Non-linear activation used in Capsules.

    Args:
        input: Input tensor.
        axis:  The dimension squash would be performed on. The default is -1
            which indicates the last dimension.

    Returns:
        A tensor of the same shape as input.
    """
    s_norm = k.sum(k.square(input), axis, keepdims=True)
    scale = s_norm / (1 + s_norm) / k.sqrt(s_norm + k.epsilon())
    return scale * input


def length(inputs):
    return k.sqrt(k.sum(k.square(inputs), -1) + k.epsilon())
