import numpy as np
from ComputeColor import *


def flowToColor(u, v, maxFlow=None):
    """
    Color code flow field U, V

    Args:
    u, v: Flow fields
    maxFlow: Optional, normalize based on this value if provided

    Returns:
    img: Color-coded flow field
    """

    maxrad = -1

    rad = np.sqrt(u**2 + v**2)
    maxrad = np.max(rad)

    u = u / maxrad * 255
    v = v / maxrad * 255

    # Compute color
    img = computeColor(u, v)

    return img
