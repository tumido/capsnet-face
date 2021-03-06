"""Facial recognition via CapsNet.

Capsule network for face recognition. Uses 3 level shallow capsule encoder
with dynamic routing by agreement and a convolutional decoder.
"""


from .dataset import preprocess, fetch_pins_people
from .network import CapsNet

__all__ = ['CapsNet', 'preprocess', 'fetch_pins_people']
