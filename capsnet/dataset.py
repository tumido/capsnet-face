import os
from zipfile import ZipFile

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils._joblib import Memory
from sklearn.utils import Bunch
from sklearn.datasets.lfw import _fetch_lfw_people
from PIL import Image
import numpy as np


PINS_DATASET = dict(
    name='frules11/pins-face-recognition',
    zip='pins-face-recognition.zip',
    folder='PINS'
)


def preprocess(people, ttt_ratio=.2, resize_to=(32,32)):
    """Preprocess data set.

    Convert `sklearn.datasets.fetch_lfw_people` output to usable data set.

    Args:
        people (sklearn.utils.Bunch): LFW data set
        ttt_ratio (float, optional): Train to test ratio. Defaults to .2.
        resize_to (tuple, optional): Target image size. Defaults to (32, 32).

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
    """

    x = people.images
    y = people.target

    def downsample(image):
        """Downsample image to `resize_to` size.

        Args:
            image (np.array): RGB Image as array

        Returns:
            np.array: RGB Image as array
        """

        image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = image.resize(resize_to, Image.ANTIALIAS)
        return np.array(image)

    x = np.array([downsample(i) for i in x]) / 255

    # Split to train and test data set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=ttt_ratio,
        # stratify=people.target,
        random_state=13
    )

    # Encode labels to one hot
    n_classes = people.target_names.shape[0]
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return (x_train, y_train), (x_test, y_test)


def dataset_gen(x, y, batch_size):
    """Training data batch generator.

    Yields batches of train data for 2 in 2 out model.

    Args:
        x (np.array): Training data.
        y (np.array): Training labels.
        batch_size (int): Size of each batch.

    Yields:
        tuple: ((x_batch, y_batch),(y_batch, x_batch))
    """

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=20,
        # horizontal_flip=True
    )

    generator = datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])


def fetch_pins_people(resize=.5, min_faces_per_person=0, color=False,
                      slice_=(slice(25, 275), slice(25, 275)),
                      download_if_missing=True):
    """Load PINS dataset.

    Use a PINS dataset provided by Kaggle, everage the scikit-learn memory
    optimizations.

    Args:
        resize (float, optional): Image resize factor. Defaults to .5.
        min_faces_per_person (int, optional): Minimal number of images per
            person. Defaults to 0.
        color (bool): Toggle is images should be in RGB or 1 channel.
            Defaults to False.
        slice_ (tuple, optional): A rectangle to which images are sliced.
            Defaults to (slice(70, 195), slice(78, 172)).
        download_if_missing (bool, optional): Set if the dataset should be
            downloaded if not present on the machine. Defaults to True.

    Returns:
        sklearn.utils.Bunch: Collection of data set
    """
    from kaggle import KaggleApi

    # Extract ZIP dataset
    kaggle_api = KaggleApi()
    kaggle_home = kaggle_api.read_config_file()['path']
    path_to_zip = os.path.join(
        kaggle_home, 'datasets', PINS_DATASET['name'], PINS_DATASET['zip']
    )
    path_to_files = os.path.join(
         kaggle_home, 'datasets', PINS_DATASET['name'], PINS_DATASET['folder']
    )

    # Download if missing
    if download_if_missing and not os.path.exists(path_to_zip):
        kaggle_api.authenticate()
        kaggle_api.dataset_download_files(PINS_DATASET['name'], quiet=False)

    if not os.path.exists(path_to_files):
        with ZipFile(path_to_zip, 'r') as zipObj:
            extraction_path = os.path.join(
                kaggle_home, 'datasets', PINS_DATASET['name']
            )
            zipObj.extractall(extraction_path)

    # Load data in memory
    m = Memory(location=kaggle_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    faces, target, target_names = load_func(
        path_to_files, resize=resize,
        min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)

    X = faces.reshape(len(faces), -1)

    # Fix names
    with np.nditer(target_names, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.core.defchararray.replace(x, 'pins ', '')
            x[...] = np.core.defchararray.replace(x, ' face', '')
            x[...] = np.core.defchararray.title(x)

    # pack the results as a Bunch instance
    return Bunch(data=X, images=faces,
                 target=target, target_names=target_names)
