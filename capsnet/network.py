import os

from keras import models, layers, callbacks as cbs, optimizers, initializers

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

    Args:
        input_shape (tuple): Input data shape - [width, height, channels]
        bins (int): Number of predicted faces
        routing_iters (int, optional): Number of iterations each routing
            should take. Defaults to 3.
        kernel_initializer (str, keras.initializer, optional): Initializer
            for routing weights. Defaults to
            initializers.random_normal(stddev=0.01, seed=0).
        init_conv_filters (int, optional): Number of filters for the first
            layer. Defaults to 256.
        init_conv_kernel (int, optional): Size of kernel for the first
            layer. Defaults to 9.
        feature_caps_kernel (int, optional): Size of kernel for Feature
            Capsules. Defaults to 5.
        feature_caps_dim (int, optional): Dimension of a capsule in
            Feature Capsule layer. Defaults to 16.
        feature_caps_channels (int, optional): Channels in each of
            capsules in Feature Capsule layer. Defaults to 16.
        prediction_caps_dim (int, optional): Dimension of each capsule in
            Prediction Capsule layer. Defaults to 32.
    """

    #pylint: disable-msg=too-many-arguments
    def __init__(
            self, input_shape, bins,
            routing_iters=3,
            kernel_initializer=initializers.random_normal(stddev=0.01, seed=0),
            init_conv_filters=256,
            init_conv_kernel=9,
            feature_caps_kernel=5,
            feature_caps_dim=16,
            feature_caps_channels=16,
            prediction_caps_dim=32
        ):
        # Input layer
        x = layers.Input(name='input_image', shape=input_shape)

        # Encoder
        conv = layers.Conv2D(
            filters=init_conv_filters,
            kernel_size=init_conv_kernel,
            strides=1,
            padding='valid',
            activation='relu',
            name='encoder_conv2d'
        )(x)
        dropout = layers.Dropout(.7, name='encoder_dropout')(conv)
        feature_caps = FeatureCapsule(
            capsule_dim=feature_caps_dim,
            channels_count=feature_caps_channels,
            kernel_size=feature_caps_kernel,
            strides=2,
            padding='valid',
            name='encoder_feature_caps'
        )(dropout)
        prediction_caps = PredictionCapsule(
            capsule_count=bins,
            capsule_dim=prediction_caps_dim,
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

        decoder = models.Sequential(
            name='decoder',
            layers=[
                layers.Dense(
                    units=400,
                    activation='relu',
                    input_dim=prediction_caps_dim*bins,
                    name='decoder_dense'
                ),
                layers.Reshape(
                    target_shape=(5, 5, 16),
                    name='decoder_reshape_1'
                ),
                layers.Lambda(
                    resize,
                    arguments=dict(target_shape=(8, 8)),
                    name='decoder_resize_1'
                ),
                layers.Conv2D(
                    4, 3,
                    activation='relu',
                    padding='same',
                    name='decoder_conv2d_1'
                ),
                layers.Lambda(
                    resize,
                    arguments=dict(target_shape=(16, 16)),
                    name='decoder_resize_2'
                ),
                layers.Conv2D(
                    8, 3,
                    activation='relu',
                    padding='same',
                    name='decoder_conv2d_2'
                ),
                layers.Lambda(
                    resize,
                    arguments=dict(target_shape=(32, 32)),
                    name='decoder_resize_3'
                ),
                layers.Conv2D(
                    16, 3,
                    activation='relu',
                    padding='same',
                    name='decoder_conv2d_3'
                ),
                layers.Conv2D(
                    3, 3,
                    activation=None,
                    padding='same',
                    name='decoder_conv2d_4'
                ),
                layers.Activation('relu', name='decoder_activation')
            ]
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
                outputs=output
            )
        )

    #pylint: disable-msg=too-many-arguments
    def train(
            self, data,
            batch_size=10,
            epochs=200,
            lr=.0001,
            lr_decay=.9,
            decoder_loss_weight=.0005,
            save_dir='model',
            extra_callbacks=None
        ):
        """Train the network.

        Args:
            data (tuple): [description]
            batch_size (int, optional): Size of a training batch.
                Defaults to 10.
            epochs (int, optional): Maximal number of epochs. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to .0001.
            lr_decay (float, optional): Learning rate decay. Defaults to .9.
            decoder_loss_weight (float, optional): Weight of decoder loss in
                total loss. Defaults to .0005.
            save_dir (str, optional): Folder name where to store training logs.
                Defaults to 'model'.
            extra_callbacks (list, optional): A list of extra callbacks for
                fit_generator. Defaults to None.

        Returns:
            keras.models.Model: A trained TensorFlow model
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
            cbs.EarlyStopping(monitor='val_loss', verbose=1, patience=20),
            *extra_callbacks
        ]

        # Compile training model
        model.compile(
            optimizer=optimizers.Adam(lr=lr),
            loss=[margin_loss, 'mse'],
            loss_weights=[1., decoder_loss_weight],
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

    def test(self, x_test, y_test, batch_size=10):
        """Test network on validation data.

        Args:
            x_test (np.array): Test inputs array
            y_test (np.array): Test labels array
            batch_size (int, optional): Size of a testing batch.
                Defaults to 10.

        Returns:
            dict: Test metrics collected as a dictionary
        """
        model = self._models['test']
        # Compile test model with the same settings
        model.compile(
            optimizer='adam',
            loss=margin_loss,
            metrics={'capsnet': 'accuracy'}
        )

        metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

        return dict(zip(model.metrics_names, metrics))

    def predict(self, x, batch_size=10):
        """Run model predictions.

        Args:
            x (np.array): Image data
            batch_size (int, optional): Size of a prediction batch.
                Defaults to 10.

        Returns:
            tuple: Prediction vector for labels and recognized feature vector
        """
        return self._models['test'].predict(x, batch_size=batch_size)

    def load_weights(self, filename):
        """Load model from a h5 file.

        Args:
            filename (str): Path to model location
        """
        print(f'Loading models\'s weights from "{filename}"...', end=' ')
        self._models['train'].load_weights(filename)
        self._models['test'].load_weights(filename, by_name=True)
        print('Done')

    def save_weights(self, filename):
        """Save model's weights.

        Args:
            filename (str): Path and filename where the model should be stored
        """
        print(f'Saving model\'s weights to "{filename}"...', end=' ')
        self._models['train'].save_weights(filename)
        print('Done')

    def summary(self):
        """Output network configuration."""
        self._models['train'].summary()
