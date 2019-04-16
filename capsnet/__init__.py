"""CapsNet.

CapsNet for face recognition.
"""


from .dataset import preprocess_lfw_people
from .network import CapsNet

__all__ = ['CapsNet', 'preprocess_lfw_people']
