import tensorflow as tf
import numpy as np

from utils.dataset import get_dataset
from utils.network import CapsNet

if __name__ == "__main__":
    data = get_dataset()

    (x_train, y_train), (x_test, y_test) = data

    capsnet = CapsNet(
        x_train.shape[1:],
        len(np.unique(y_train, axis=0))
    )

    capsnet.models['train'].summary()
    capsnet.models['inference'].summary()
    capsnet.train(data)
