import numpy as np
from sklearn.datasets import fetch_lfw_people

from capsnet import preprocess_lfw_people, CapsNet

if __name__ == "__main__":
    people = fetch_lfw_people(
        color=True,
        min_faces_per_person=25,
        # resize=1.,
        # slice_=(slice(48, 202), slice(48, 202))
    )

    data = preprocess_lfw_people(people)

    (x_train, y_train), (x_test, y_test) = data

    capsnet = CapsNet(
        x_train.shape[1:],
        len(np.unique(y_train, axis=0))
    )

    capsnet.models['train'].summary()
    capsnet.train(data, batch_size=10)
