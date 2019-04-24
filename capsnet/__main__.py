import numpy as np
from sklearn.datasets import fetch_lfw_people
from keras import callbacks

from capsnet import preprocess_lfw_people, CapsNet

def main():
    """CapsNet run as module.

    Run full cycle when CapsNet is run as a module.
    """
    people = fetch_lfw_people(
        color=True,
        min_faces_per_person=25,
        # resize=1.,
        # slice_=(slice(48, 202), slice(48, 202))
    )

    data = preprocess_lfw_people(people)

    (x_train, y_train), (x_test, y_test) = data  # noqa

    capsnet = CapsNet(
        x_train.shape[1:],
        len(np.unique(y_train, axis=0))
    )

    capsnet.models['train'].summary()

    # Start TensorBoard
    tb = callbacks.TensorBoard(
        'model/tensorboard_logs', batch_size=10,
        histogram_freq=1, write_graph=True, write_grads=True,
        write_images=True
    ),
    capsnet.train(data, batch_size=10, extra_callbacks=[tb])

if __name__ == "__main__":
    main()
