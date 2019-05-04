"""Layers used by CapsNet."""
import tensorflow as tf
from keras import layers, initializers, backend as k

from .activations import squash


class PredictionCapsule(layers.Layer):
    """PredictionCapsule layer.

    A prediction capsule with dynamic routing. A placement above feature
    capsule layer is expected. Extends <keras.layers.Layer>.

    Args:
        capsule_count (int): Number of capsules in this layer
        capsule_dim (int): Dimensionality of each capsule
        kernel_initializer (str, keras.initializers, optional): Weight
            initializer generator. Defaults to
            `keras.initializers.random_normal`.
        routing_iters (int, optional): Number of iterations for each routing.
            Defaults to 3.
        kwargs (dict): Additional parameters passed to layer. Required to
            maintain compatibility.
    """

    def __init__(self, capsule_count, capsule_dim,
                 kernel_initializer=None, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.capsule_count = capsule_count
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters

        if isinstance(kernel_initializer, str):
            kernel_initializer = initializers.get(kernel_initializer)
        if not kernel_initializer:
            kernel_initializer = \
                initializers.random_normal(stddev=0.01, seed=0)
        self.kernel_initializer = kernel_initializer

        # Define feature variables
        self.input_capsule_count = None
        self.W = None

        super(PredictionCapsule, self).__init__(**kwargs)

    def get_config(self):
        """Return the config of the layer.

        Extends <keras.layers.Layer.get_config>
        """
        return dict(
            capsule_count=self.capsule_count,
            capsule_dim=self.capsule_dim,
            routing_iters=self.routing_iters,
            kernel_initializer=self.kernel_initializer,
            **super(PredictionCapsule, self).get_config()
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
        assert len(input_shape) == 3, (
            'The input Tensor shape must be '
            '[None, input_capsule_count, input_capsule_dimension]'
        )
        self.input_capsule_count = input_shape[1]
        input_capsule_dim = input_shape[2]

        # Define transformation matrix
        self.W = self.add_weight(
            shape=[
                self.capsule_count, self.input_capsule_count,
                self.capsule_dim, input_capsule_dim
            ],
            name='W',
            initializer=self.kernel_initializer,
            trainable=True
        )

        super(PredictionCapsule, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Layer logic.

        Overrides <keras.layers.Layers.call>

        Args:
            inputs: Input tensor.

        Returns:
            A tensor.
        """
        # Prepare inputs
        # inputs == [None, input_capsule_count, input_capsule_dim]
        u = k.expand_dims(inputs, 1)
        # u == [None, 1, input_capsule_count, input_capsule_dim]
        u = k.tile(u, (1, self.capsule_count, 1, 1))
        # u == [None, capsule_count, input_capsule_count, input_capsule_dim]

        # Perform: inputs x W by scanning on input[0]
        u = tf.einsum('iabc,abdc->iabd', u, self.W)
        # u == [None, capsule_count, input_capsule_count, capsule_dim]

        # Routing
        # Init log prior probabilities to zeros:
        b = tf.zeros(shape=(
            k.shape(inputs)[0], self.capsule_count,
            self.input_capsule_count, 1
        ))
        # b == [None, capsule_count, input_capsule_count, 1]

        for i in range(self.routing_iters):
            with tf.variable_scope(f'routing_{i}'):
                c = tf.keras.activations.softmax(b, axis=1)
                # c == [None, capsule_count, input_capsule_count, 1]
                # Perform: sum(c x u)
                #
                # c == [None, capsule_count, input_capsule_count, 1]
                # u == [None, capsule_count, input_capsule_count, capsule_dim]
                s = tf.reduce_sum(tf.multiply(c, u), axis=2, keepdims=True)
                # s == [None, capsule_count, 1, capsule_dim]
                # Perform: squash
                v = squash(s)
                # v == [None, capsule_count, 1, capsule_dim]

                # Perform: sum(output x input)
                v_tiled = tf.tile(v, (1, 1, self.input_capsule_count, 1))
                b += tf.reduce_sum(
                    tf.matmul(u, v_tiled, transpose_b=True),
                    axis=3, keepdims=True
                )
                # b == [None, capsule_count, input_capsule_count, 1]

        # Squeeze the extra dim (used for manipulation, not needed on output)
        # v == [None, capsule_count, 1, capsule_dim]
        v = tf.squeeze(v, axis=2)
        # v == [None, capsule_count, capsule_dim]
        return v


def FeatureCapsule(capsule_dim, channels_count,  # noqa
                   kernel_size, strides, padding, name=''):
    """FeatureFace capsule layer.

    Args:
        capsule_dim: Dimension of the output vector of each capsule
        channels_count: Types of capsules
        kernel_size: Param for Conv2D
        strides: Param for Conv2D
        padding: Param for Conv2D

    Returns:
        func: Composite keras layer
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


class Mask(layers.Layer):
    """Masking layer.

    Layer used to combine 2 inputs into a single output. Masks tensor passed
    as the first input with the second tensor. Second tensor should contain
    hot one encoding of desired activations.
    """
    def call(self, inputs, **kwargs):
        """Layer logic.

        Overrides <keras.layers.Layers.call>

        Args:
            inputs: Input tensor.

        Returns:
            A tensor.
        """
        capsule_output, labels = inputs

        return k.batch_flatten(capsule_output * k.expand_dims(labels))

    def compute_output_shape(self, input_shape):
        """Layer output shape.

        Args:
            input_shape (list): List of input shapes

        Returns:
            tuple: Output shape for this layer
        """
        return (None, input_shape[0][1] * input_shape[0][2])
