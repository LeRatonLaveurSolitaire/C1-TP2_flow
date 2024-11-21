import numpy as np


def computeColor(u, v):
    """
    Compute color coding for flow field U, V
    """
    nanIdx = np.isnan(u) | np.isnan(v)
    u = np.where(nanIdx, 0, u)
    v = np.where(nanIdx, 0, v)

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1

    f = fk - k0

    img = np.zeros((*u.shape, 3), dtype=np.uint8)

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75

        img[:, :, i] = np.floor(255 * col * (1 - nanIdx)).astype(np.uint8)

    return img


def makeColorwheel():
    """
    Generate color wheel for optical flow visualization
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY

    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC

    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col : col + MR, 0] = 255

    return colorwheel.astype(np.uint8)
