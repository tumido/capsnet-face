# CapsNet classifier for LFW
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/license-APACHE2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

A CapsNet network implementing a facial recognition for a LFW and similar data sets.

## Generic usage

### Installation

The provided `capsnet` package can be easily installed via PIP:

```python
pip install git+https://github.com/tumido/capsnet-face
```

The package provides binding to a [Kaggle](https://www.kaggle.com) data set, if you desire use it as your data source please install `kaggle` package as well and setup accordingly.

### Local setup

If you want to create a developer setup for this package, you can either use Conda or Pipenv to do so:

```bash
$ git clone https://github.com/tumido/capsnet-face
Cloning into 'capsnet-face'...
remote: Enumerating objects: 95, done.
remote: Counting objects: 100% (95/95), done.
remote: Compressing objects: 100% (67/67), done.
remote: Total 330 (delta 52), reused 69 (delta 28), pack-reused 235
Receiving objects: 100% (330/330), 2.02 MiB | 4.12 MiB/s, done.
Resolving deltas: 100% (200/200), done.
$ cd capsnet-face
```

Then sync via Pipenv:
```bash
$ pipenv sync --dev
Creating a virtualenv for this project…
...
⠏ Creating virtual environment...
...
Installing dependencies from Pipfile.lock (2dc3d7)…
  🐍   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 96/96 — 00:01:15
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
All dependencies are now up-to-date!

$ pipenv shell
Launching subshell in virtual environment…
...
```

Or use Anaconda/Conda:

```bash
$ conda env create -f environment.yml
Collecting package metadata: done
Solving environment: done
Preparing transaction: done
...

$ conda activate keras_cpu
...
```

Python 3.6 supported.

### CLI

The package provides a CLI interface:

```python
$ capsnet --help
Using TensorFlow backend.
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  CapsNet for facial recognition. CLI interface.

Options:
  --help  Show this message and exit.

Commands:
  predict  Predict a label for input image
  train    Train your model.
```

To invoke training please do:

```bash
$ capsnet train -d lfw
Using TensorFlow backend.
Loading data set: LFW
Creating CapsNet network...
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_image (InputLayer)        (None, 32, 32, 3)    0
__________________________________________________________________________________________________
encoder_conv2d (Conv2D)         (None, 24, 24, 256)  62464       input_image[0][0]
__________________________________________________________________________________________________
encoder_dropout (Dropout)       (None, 24, 24, 256)  0           encoder_conv2d[0][0]
__________________________________________________________________________________________________
encoder_feature_caps_conv2d (Co (None, 10, 10, 256)  1638656     encoder_dropout[0][0]
__________________________________________________________________________________________________
encoder_feature_caps_reshape (R (None, 1600, 16)     0           encoder_feature_caps_conv2d[0][0]
__________________________________________________________________________________________________
encoder_feature_caps_squash (La (None, 1600, 16)     0           encoder_feature_caps_reshape[0][0
__________________________________________________________________________________________________
encoder_pred_caps (PredictionCa (None, 42, 32)       34406400    encoder_feature_caps_squash[0][0]
__________________________________________________________________________________________________
input_label (InputLayer)        (None, 42)           0
__________________________________________________________________________________________________
mask (Mask)                     (None, 1344)         0           encoder_pred_caps[0][0]
                                                                 input_label[0][0]
__________________________________________________________________________________________________
capsnet (Lambda)                (None, 42)           0           encoder_pred_caps[0][0]
__________________________________________________________________________________________________
decoder (Sequential)            (None, 32, 32, 3)    540479      mask[0][0]
==================================================================================================
Total params: 36,647,999
Trainable params: 36,647,999
Non-trainable params: 0
__________________________________________________________________________________________________

Starting training...
Epoch 1/200
...
```

This launches the standard training routine over LFW dataset. Kaggle PINS dataset is also available

Predictions can be run directly using a saved model:

```bash

$ predict -m saved_models/2019-05-14_11-caps_75-acc.tar.gz Serena_Williams_0002.jpg
Loading CapsNet...
Loading model from saved_models/2019-05-14_11-caps_75-acc.tar.gz...
        Loading "train" architecture... Done
        Loading "test" architecture... Done
        Loading weights... Done
        Extracting labels... Done
Loading image...
Predicting...
Guessed likelihood per label:
Label               Probability
Ariel Sharon             2.83%
Colin Powell             8.25%
Donald Rumsfeld          3.25%
George W Bush            0.79%
Gerhard Schroeder        0.25%
Hugo Chavez              0.08%
Jacques Chirac           0.08%
Jean Chretien            0.11%
John Ashcroft            0.65%
Junichiro Koizumi        0.16%
Serena Williams         91.07% <-- Best match
Tony Blair               0.23%
```

### Manual

When you desire to experiment with this implementation, please feel free to do so by importing the `CapsNet` class from the installed package. We encourage to explore `help(CapsNet)` to list all methods and arguments possible.

This package provides multiple objects:

- `CapsNet`, the network class
- `preprocess` function for input data preprocessing
- `fetch_pins_people` a Kaggle [PINS](https://www.kaggle.com/frules11/pins-face-recognition) data set collector with API consistent to the [`fetch_lfw_people`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)

## Model evaluation

There is a [notebooks](notebooks) folder available in this repository. If you would like to know more about the process how the models were trained or evaluated, please feel free to explore these.

### Collected accuracies over multiple runs

| Identities | Data set (images) | Routing iterations | Acc. Train | Acc. Validation | Acc. Test  | Loss   |
| ---------: | ----------------: | -----------------: | ---------: | --------------: | ---------: | -----: |
|         42 |        LFW (2588) |                  1 |      46.2% |           42.5% |      42.5% | 0.5002 |
|         42 |        LFW (2588) |                  3 |      56.4% |           53.7% |      42.5% | 0.3915 |
|         11 |        LFW (1560) |                  1 |      52.6% |           63.2% |      61.5% | 0.2952 |
|         11 |        LFW (1560) |                  3 |      69.3% |           75.0% |      73.7% | 0.2013 |

The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1Ym8p-9WcOMwvHaDBS5LmgdqeDUNxzk3F?usp=sharing).

### Visualized activations

![George W. Bush](images/predicted_131_fail.svg)
![George W. Bush](images/predicted_116_ok.svg)
![Gloria Macapagal Arroyo](images/predicted_218_ok.svg)
![Serena Williams](images/predicted_316_ok.svg)
![Junichiro Koizumi](images/predicted_255_ok.svg)

### Confusion matrix for the 11 identities model

![Confusion matrix](images/confusion.svg)
