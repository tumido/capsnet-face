import os

from keras import models, layers, callbacks as cbs, optimizers

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

    def __init__(self, input_shape, bins):
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
        feature_caps = FeatureCapsule(
            capsule_dim=8,
            channels_count=32,
            kernel_size=9,
            strides=2,
            padding='valid',
            name='encoder_feature_caps'
        )(conv)
        prediction_caps = PredictionCapsule(
            capsule_count=bins,
            capsule_dim=16,
            routing_iters=3,
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
                input_dim=16*bins,
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
                )
            )
        )
        decoder.add(
            layers.Conv2D(
                4,
                9,
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
                )
            )
        )
        decoder.add(
            layers.Conv2D(
                16,
                9,
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
                )
            )
        )
        decoder.add(
            layers.Conv2D(
                3,
                9,
                activation=None,
                padding='same',
                name='decoder_conv2d_3'
            )
        )
        decoder.add(
            layers.Reshape(
                target_shape=input_shape,
                name='decoder_reshape_2'
            )
        )

        masked_train = Mask(name='mask_with_labels')([prediction_caps, y])
        masked_inference = Mask(name='mask')(prediction_caps)

        # Models
        self.models = dict(
            train=models.Model(
                inputs=[x, y],
                outputs=[output, decoder(masked_train)]
            ),
            inference=models.Model(
                inputs=x,
                outputs=[feature_caps, output, decoder(masked_inference)]
            )
        )

    #pylint: disable-msg=too-many-arguments
    def train(self, data, batch_size=10, epochs=100,
              lr=.0001, lr_decay=.9, decoder_loss_weight=.0005,
              save_dir='model', callbacks=[]):
        """Train the network.

        Args:
            data (tuple): Tuple containing train and test data along with their labels

        Returns:
            A trained TensorFlow model

        """
        (x_train, y_train), (x_test, y_test) = data

        # Ensure model directory
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Callback
        cb = [
            cbs.CSVLogger(f'{save_dir}/log.csv'),
            cbs.LearningRateScheduler(lambda e: lr * (lr_decay ** e)),
            cbs.ModelCheckpoint(
                f'{save_dir}/weights.{{epoch:02d}}.h5', 'val_capsnet_acc',
                save_best_only=True, save_weights_only=True, verbose=1
            ),
            cbs.TensorBoard(
                f'{save_dir}/tensorboard_logs', batch_size=batch_size
            ),
            *callbacks
        ]

        self.models['train'].compile(
            optimizer=optimizers.Adam(lr=lr),
            loss=[margin_loss, 'mse'],
            loss_weights=[1., decoder_loss_weight],
            metrics={'capsnet': 'accuracy'}
        )

        hist = self.models['train'].fit_generator(
            generator=dataset_gen(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            epochs=epochs,
            validation_data=[[x_test, y_test], [y_test, x_test]],
            verbose=1,
            callbacks=cb
        )

        self.models['train'].save_weights(f'{save_dir}/model.h5')

        return hist

    def inference(self, data):
        pass
