from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


def get_dataset(min_faces_per_person=25, ttt_ratio=.2):
    people = fetch_lfw_people(
        color=True,
        min_faces_per_person=min_faces_per_person,
        # resize=1.0,
        # slice_=(slice(48, 202), slice(48, 202))
    )

    x_train, x_test, y_train, y_test = train_test_split(
        people.images,
        people.target,
        test_size=ttt_ratio
    )

    n_classes = people.target_names.shape[0]
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return (x_train, y_train), (x_test, y_test)


def dataset_gen(x, y, batch_size=32):
    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True
    )

    generator = datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])

