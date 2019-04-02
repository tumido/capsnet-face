"""Layers used by CapsNet."""
import tensorflow as tf
import keras.backend as k
from keras import layers, initializers

from .activations import squash


class Capsule(layers.Layer):
    """Capsule layer."""

    def __init__(self, capsule_count, capsule_dim, routing_iters=3, **kwargs):
        """Capsule layer."""
        super().__init__(**kwargs)
        self.capsule_count = capsule_count
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters

        # ? Init to zeros?
        self.input_capsule_count = None

    def get_config(self):
        """Return the config of the layer.

        Extends <keras.layers.Layer.get_config>
        """
        return dict(
            capsule_count=self.capsule_count,
            capsule_dim=self.capsule_dim,
            routing_iters=self.routing_iters,
            **super().get_config()
        )


    def compute_output_shape(self, _):
        """Compute the output shape of the layer.

        Overrides <keras.layers.Layer.compute_output_shape>
        """
        return (None, self.capsule_count, self.capsule_dim)

    def build(self, input_shape):
        """Create the variables of the layer.

        Overrides <keras.layers.Layer.build>
        """
        tf.assert_rank(
            input_shape,
            rank=3,
            message=(
                'The input Tensor shape must be '
                '[None, input_capsule_count, input_capsule_dimension]'
            )
        )
        self.input_capsule_count = input_shape[1]
        input_capsule_dim = input_shape[2]

        # Define transformation matrix
        self.W = self.add_weight(
            name='W',
            initializer=initializers.glorot_uniform,
            shape=[
                self.capsule_count, self.input_capsule_count,
                self.capsule_dim, input_capsule_dim
            ]
        )

        super().build(input_shape)

    def call(self, inputs):
        """Layer logic.

        Overrides <keras.layers.Layers.call>

        Args:
            inputs: Input tensor.

        Returns:
            A tensor.
        """
        # Prepare inputs
        # inputs.shape == [None, input_capsule_count, input_capsule_dim]
        inputs = k.expand_dims(inputs, 1)
        # inputs.shape == [None, 1, input_capsule_count, input_capsule_dim]
        inputs = k.tile(inputs, (1, self.capsule_count, 1, 1))
        # inputs.shape == [None, capsule_count, input_capsule_count, input_capsule_dim]

        # Perform: inputs x W by scanning on input[0]
        # Treat first 2 dimensions as batch dimensions
        # [input_capsule_dim] x [capsule_dim, input_capsule_dim]
        inputs = k.map_fn(lambda x: k.batch_dot(x, self.W, [2, 3]), inputs)
        # inputs.shape == [None, capsule_dim, input_capsule_dim, capsule_dim]

        # Routing
        # Init log prior probabilities to zeros:
        b = tf.zeros(shape=(k.shape(inputs)[0], self.capsule_count, self.input_capsule_count))
        # b.shape == [None, capsule_count, input_capsule_count]

        for i in range(self.routing_iters):
            c = tf.keras.activations.softmax(b, axis=1)
            # c.shape == [batch_size, capsule_count, input_capsule_count]
            # Perform: c x input
            # Treat first 2 dimensions of each tensor as batch dimensions and perform dot:
            # [input_capsule_count] x [input_capsule_count, capsule_dim]
            # Perform: squash
            outputs = squash(k.batch_dot(c, inputs, [2, 2]))
            # output.shape == [None, capsule_count, capsule_dim]

            # Update b only if not the last iteration
            if i == self.routing_iters-1:
                # Perform: output x input
                # Treat first 2 dimensions as batch dimensions and dot the rest:
                # [capsule_dim] x [input_capsule_count, capsule_dim]
                b += k.batch_dot(outputs, inputs, [2, 3])
                # b.shape == [None, capsule_count, input_capsule_count]

        return outputs


def PrimaryCapsule(capsule_dim, channels_count,
                   kernel_size, strides, padding, name=''):
    """Primary capsule layer.

    Args:
        capsule_dim: Dimension of the output vector of each capsule
        channels_count: Types of capsules
        kernel_size: Param for Conv2D
        strides: Param for Conv2D
        padding: Param for Conv2D
    """
    def _layer(inputs):
        """Primary capsule layer

        Args:
            inputs: Input tensor

        Returns:
            A tensor
        """

        # Apply Conv2D for each channel
        outputs = layers.Conv2D(
            capsule_dim*channels_count,
            kernel_size,
            strides=strides,
            padding=padding,
            name=f'{name}_conv2d'
        )(inputs)

        # Concatenate all capsules
        outputs = layers.Reshape(
            [-1, capsule_dim],
            name=f'{name}_reshape'
        )(outputs)

        outputs = layers.Lambda(
            squash,
            name=f'{name}_squash'
        )(outputs)

        return outputs
    return _layer
