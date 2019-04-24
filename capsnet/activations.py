"""Activation functions used by CapsNet."""

import keras.backend as k
import tensorflow as tf


def squash(inputs, axis=-1):
    """Non-linear activation used in Capsules.

    Args:
        inputs: Input tensor.
        axis:  The dimension squash would be performed on. The default is -1
            which indicates the last dimension.

    Returns:
        A tensor of the same shape as input.
    """
    inputs += k.epsilon()
    s_norm = k.sum(k.square(inputs), axis, keepdims=True)
    scale = s_norm / (1 + s_norm) / k.sqrt(s_norm)
    return scale * inputs


def length(inputs):
    """Compute length from a tensor.

    Used as layer.Lambda function to provide auxilary output of CapsNet.
    Instead of a Capsule its length is used.

    Args:
        inputs: Input tensor

    Returns:
        A tensor of shape (None, num_capsules)
    """
    return k.sqrt(k.sum(k.square(inputs), axis=2))


def resize(inputs, target_shape):
    """Resize image.

    Used as layers.Lambda function to reshape a tensor image to enhance
    dimensions.

    Args:
        inputs: Input tensor
        target_shape: Desired shape (HEIGHT, WIDTH)

    Returns:
        A tensor of shape (None, HEIGHT, WIDTH)
    """
    return tf.image.resize_nearest_neighbor(inputs, target_shape)
