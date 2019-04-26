import os

from keras import models, layers, callbacks as cbs, optimizers, initializers
import numpy as np

from .layers import Mask, PredictionCapsule, FeatureCapsule
from .losses import margin_loss
from .dataset import dataset_gen
from .activations import length, resize


class CapsNet:
    """Capsule neural network for Face Recognition.

    A neural network using a CapsNet architecture with capsules mapped to each
    biometric property which wetend to recognise. Produces an identity tensor
    for each given image, where each member is corresponding to configuration
    and weights in capsules.
    """

    def __init__(
            self, input_shape, bins, routing_iters=3,
            kernel_initializer=initializers.random_normal(stddev=0.01, seed=0)
    ):
        """CapsNet instance constructor.

        Args:
            input_shape: Input data shape - [width, height, channels]
            bins: Number of predicted faces

        """
        # Input layer
        x = layers.Input(name='input_image', shape=input_shape)

        # Encoder
        conv = layers.Conv2D(
            filters=256,
            kernel_size=9,
            strides=1,
            padding='valid',
            activation='relu',
            name='encoder_conv2d'
        )(x)
        dropout = layers.Dropout(.7, name='encoder_dropout')(conv)
        feature_caps = FeatureCapsule(
            capsule_dim=16,
            channels_count=16,
            kernel_size=5,
            strides=2,
            padding='valid',
            name='encoder_feature_caps'
        )(dropout)
        prediction_caps = PredictionCapsule(
            capsule_count=bins,
            capsule_dim=32,
            routing_iters=routing_iters,
            kernel_initializer=kernel_initializer,
            name='encoder_pred_caps'
        )(feature_caps)

        output = layers.Lambda(
            length,
            name='capsnet'
        )(prediction_caps)

        # Decoder
        y = layers.Input(name='input_label', shape=(bins,))

        decoder = models.Sequential(name='decoder')
        decoder.add(
            layers.Dense(
                units=400,
                activation='relu',
                input_dim=32,
                name='decoder_dense'
            )
        )
        decoder.add(
            layers.Reshape(
                target_shape=(5, 5, 16),
                name='decoder_reshape_1'
            )
        )
        decoder.add(
            layers.Lambda(
                resize,
                arguments=dict(
                    target_shape=(8, 8)
                ),
                name='decoder_resize_1'
            )
        )
        decoder.add(
            layers.Conv2D(
                4,
                3,
                activation='relu',
                padding='same',
                name='decoder_conv2d_1'
            )
        )
        decoder.add(
            layers.Lambda(
                resize,
                arguments=dict(
                    target_shape=(16, 16)
                ),
                name='decoder_resize_2'
            )
        )
        decoder.add(
            layers.Conv2D(
                8,
                3,
                activation='relu',
                padding='same',
                name='decoder_conv2d_2'
            )
        )
        decoder.add(
            layers.Lambda(
                resize,
                arguments=dict(
                    target_shape=(32, 32)
                ),
                name='decoder_resize_3'
            )
        )
        decoder.add(
            layers.Conv2D(
                16,
                3,
                activation='relu',
                padding='same',
                name='decoder_conv2d_3'
            )
        )
        decoder.add(
            layers.Conv2D(
                3,
                3,
                activation=None,
                padding='same',
                name='decoder_conv2d_4'
            )
        )
        decoder.add(
            layers.Activation('sigmoid', name='decoder_activation')
        )

        masked = Mask(name='mask')([prediction_caps, y])

        # Models
        self._models = dict(
            train=models.Model(
                inputs=[x, y],
                outputs=[output, decoder(masked)]
            ),
            test=models.Model(
                inputs=x,
                outputs=[output, feature_caps]
            )
        )

    #pylint: disable-msg=too-many-arguments
    def train(self, data, batch_size=10, epochs=100,
              lr=.0001, lr_decay=.9, decoder_loss_weight=.0005,
              save_dir='model', extra_callbacks=None):
        """Train the network.

        Args:
            data (tuple): Tuple containing train and test data along with their labels

        Returns:
            A trained TensorFlow model

        """
        (x_train, y_train), (x_test, y_test) = data
        model = self._models['train']

        # Ensure model directory
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Callback
        extra_callbacks = extra_callbacks if extra_callbacks else []
        cb = [
            cbs.CSVLogger(f'{save_dir}/log.csv'),
            cbs.LearningRateScheduler(lambda e: lr * (lr_decay ** e)),
            cbs.ModelCheckpoint(
                f'{save_dir}/weights.{{epoch:02d}}.h5', 'val_capsnet_acc',
                save_best_only=True, save_weights_only=True, verbose=1
            ),
            *extra_callbacks
        ]

        # Compile training model
        model.compile(
            optimizer=optimizers.Adam(lr=lr),
            loss=[margin_loss, 'mse'],
            loss_weights=[1., decoder_loss_weight],
            metrics={'capsnet': 'accuracy'}
        )

        # Compile test model with the same settings
        self._models['test'].compile(
            optimizer=optimizers.Adam(lr=lr),
            loss=margin_loss,
            metrics={'capsnet': 'accuracy'}
        )

        # Execute training
        return model.fit_generator(
            generator=dataset_gen(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            epochs=epochs,
            validation_data=[[x_test, y_test], [y_test, x_test]],
            verbose=1,
            callbacks=cb
        )

    def test(self, x_test, y_test):
        """Test network on validation data

        Args:
            data (tuple): Tuple contaning test data with respective labels

        Returns:
            tuple: Precition vector for labels
        """
        return self._models['test'].evaluate(x_test, y_test)

    def predict(self, x):
        """Run model predictions

        Args:
            image (np.array): Image data

        Returns:
            tuple: Prediction vector for labels and recognized feature vector
        """
        return self._models['test'].predict(images)

    def load_weights(self, filename):
        """Load model from a h5 file

        Args:
            filename (str): Path to model location
        """
        self._models['train'].load_weights(filename)

    def save_weights(self, filename):
        """Save model's weights

        Args:
            filename (str): Path and filename where the model should be stored
        """
        self._models['train'].save_weights(filename)

    def summary(self):
        """Output network configuration"""
        self._models['train'].summary()
