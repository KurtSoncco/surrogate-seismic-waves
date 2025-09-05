"""
Surrogate Seismic Waves SDK
"""

from .acc2fas import acceleration_to_fas
from .kohmachi import kohmachi
from .ttf import TTF

__all__ = [
    "acceleration_to_fas",
    "kohmachi",
    "TTF",
]
