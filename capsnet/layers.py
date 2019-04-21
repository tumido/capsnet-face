"""Layers used by CapsNet."""
import tensorflow as tf
import keras.backend as k
from keras import layers, initializers

from .activations import squash


class PredictionCapsule(layers.Layer):
    """PredictionCapsule layer."""

    def __init__(self, capsule_count, capsule_dim,
                 kernel_initializer, routing_iters=3, **kwargs):
        """Capsule layer."""
        super().__init__(**kwargs)
        self.capsule_count = capsule_count
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters

        self.kernel_initializer = kernel_initializer
        if isinstance(kernel_initializer, str):
            self.kernel_initializer = initializers.get(kernel_initializer)

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
        with tf.name_scope(self.name):
            tf.summary.histogram('W', self.W)

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
            with tf.variable_scope(f'routing_{i}'):
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


def FeatureCapsule(capsule_dim, channels_count,  # noqa
                   kernel_size, strides, padding, name=''):
    """FeatureFace capsule layer.

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


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs, mask = inputs
        else:
            lengths = k.sqrt(k.sum(k.square(inputs), -1))
            indices = k.argmax(lengths, 1)
            num_classes = lengths.get_shape().as_list()[1]
            mask = k.one_hot(indices, num_classes)
        return k.batch_flatten(inputs * k.expand_dims(mask, -1))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):
            input_shape = input_shape[0]
        return (None, input_shape[1] * input_shape[2])
