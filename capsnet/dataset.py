from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


def preprocess_lfw_people(people, ttt_ratio=.2):
    """Preprocess data set.

    Convert `sklearn.datasets.fetch_lfw_people` output to usable data set.

    Args:
        people (sklearn.utils.Bunch): LFW data set
        ttt_ratio (float, optional): Train to test ratio. Defaults to .2.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
    """

    x = people.images
    y = people.target

    def downsample(image):
        """Downsample image to 32x32.

        Args:
            image (np.array): RGB Image as array

        Returns:
            np.array: RGB Image as array
        """

        image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = image.resize((32, 32), Image.ANTIALIAS)
        return np.array(image)

    x = np.array([downsample(i) for i in x]) / 255

    # Split to train and test data set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=ttt_ratio,
        # stratify=people.target,
        # random_state=54
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
        # rotation_range=20,
        horizontal_flip=True
    )

    generator = datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
