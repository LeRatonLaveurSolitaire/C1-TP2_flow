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
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    height, width = u.shape

    maxu = -999
    maxv = -999
    minu = 999
    minv = 999
    maxrad = -1

    # Fix unknown flow
    idxUnknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxrad, np.max(rad))

    print(
        f"max flow: {maxrad:.4f} flow range: u = {minu:.3f} .. {maxu:.3f}; v = {minv:.3f} .. {maxv:.3f}"
    )

    if maxFlow is not None:
        if maxFlow > 0:
            maxrad = maxFlow

    eps = np.finfo(float).eps
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    # Compute color
    img = computeColor(u, v)

    # Set unknown flow to black
    img[np.repeat(idxUnknown[:, :, np.newaxis], 3, axis=2)] = 0

    return img
