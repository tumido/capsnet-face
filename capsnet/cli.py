import os

import click
import numpy as np
import yaml

import tensorflow as tf
from .network import CapsNet
from .dataset import fetch_pins_people, preprocess, downsample
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets.lfw import _load_imgs

DATASETS = dict(
    lfw=fetch_lfw_people,
    pins=fetch_pins_people
)

@click.group()
def cli():
    """CapsNet for facial recognition. CLI interface."""
    pass

@cli.command()
@click.option(
    '-d', '--dataset',
    type=click.Choice(['lfw', 'pins']),
    required=True,
    help='Specify dataset.'
)
@click.option(
    '--min',
    default=25,
    help='Minimal amount of images per label.'
)
@click.option(
    '--save-to',
    default='/tmp',
    type=click.Path(exists=True),
    help='Store the trained model in this directory.'
)
def train(dataset, min, save_to):
    """Train your model."""
    click.echo(click.style(f'Loading data set: {dataset.upper()}', fg='green'))
    dataset_func = DATASETS[dataset]
    people = dataset_func(
        color=True,
        min_faces_per_person=min,
    )
    data = preprocess(people)
    (x_train, y_train), (x_test, y_test) = data  # noqa

    click.echo(click.style('Creating CapsNet network...', fg='green'))
    model = CapsNet(
        x_train.shape[1:],
        len(np.unique(y_train, axis=0))
    )

    click.echo(model.summary())

    click.echo(click.style('Starting training...', fg='green'))
    model.train(data, batch_size=10)

    click.echo('Saving model...')
    labels = list(people.target_names)
    model.save(save_to, labels)

    click.echo('Testing...')
    metrics = model.test(x_test, y_test)
    click.echo(pprint(metrics))


@cli.command()
@click.option(
    '-m', '--model',
    type=click.Path(exists=True),
    required=True,
    help='Existing model tar.gz location.'
)
@click.argument(
    'image',
    type=click.Path(exists=True)
)
def predict(model, image):
    """Predict a label for input image"""
    # Suppress Keras deprecation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    yaml.warnings({'YAMLLoadWarning': False})
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load network and labels
    click.echo(click.style('Loading CapsNet...', fg='green'))
    model, labels = CapsNet.load(model)

    # Load the image (default slice for LFW)
    click.echo(click.style('Loading image...', fg='green'))
    images = _load_imgs(
        [image],
        slice_=(slice(70, 195), slice(78, 172)),
        color=True,
        resize=0.5
    )
    images = np.array([downsample(i, (32,32)) for i in images]) / 255

    click.echo(click.style('Predicting...', fg='green'))
    prediction = model.predict(images)[0]

    # Print a table with predictions
    click.echo(click.style('Guessed likelihood per label:', fg='green'))

    click.echo(click.style(f'{"Label":20}{"Probability":10}', fg='blue'))
    best = np.argmax(prediction)
    for idx, (l, p) in enumerate(zip(labels, prediction)):
        msg = f'{l:20}{p:>10.2%}'
        if idx == best:
            click.echo(click.style(f'{msg} <-- Best match', fg='green'))
        else:
            click.echo(msg)

if __name__ == "__main__":
    cli()
