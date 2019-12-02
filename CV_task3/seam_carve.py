import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
from copy import deepcopy
from collections.abc import Iterable


def get_grad(brightness):
    y_grad = np.zeros(brightness.shape)
    y_grad[0] = brightness[1] - brightness[0]
    y_grad[-1] = brightness[-1] - brightness[-2]
    y_grad[1:-1] = brightness[2:] - brightness[:-2]
    return y_grad

def get_energy(brightness, mask):
    sh = brightness.shape
    y_grad = get_grad(brightness)
    x_grad = get_grad(brightness.T).T
    energy = np.sqrt(x_grad ** 2 + y_grad ** 2)

    val = sh[0] * sh[1] * 256
    if mask is not None:
        energy += mask * val
    return energy

def get_seam(energy):
    sh = energy.shape
    seam = deepcopy(energy)
    for i in range(1, sh[0]):
        seam[i, 0] = min(seam[i - 1, 0], seam[i - 1, 1]) + energy[i, 0]
        seam[i, -1] = min(seam[i - 1, -1], seam[i - 1, -2]) + energy[i, -1]
        for j in range(1, sh[1] - 1):
            seam[i, j] = min(seam[i - 1, j - 1], seam[i - 1, j + 1], seam[i - 1, j]) + energy[i, j]
    return seam

def shrink(energy):
    seam = get_seam(energy)
    sh = energy.shape
    mask = np.zeros(sh, dtype=float)
    mi = [np.inf, -1]
    for j_diff in range(sh[1]):
        if mi[0] > seam[sh[0] - 1, j_diff]:
            mi = [seam[sh[0] - 1, j_diff], j_diff]
    j = mi[1]
    for i in range(sh[0] - 1, -1, -1):
        mask[i, j] = 1
        if i == 0:
            continue
        mi = [np.inf, -1]
        for j_diff in range(max(0, j - 1), min(sh[1], j + 2)):
            if mi[0] > seam[i - 1, j_diff]:
                mi = [seam[i - 1, j_diff], j_diff]
        j = mi[1]
        #here we get only 0, 1 or 2, so need to add min value
    return mask, mask, mask

def expand(energy):
    return shrink(energy)

def seam_carve(img, mode, mask):
    direction, method = mode.split()
    transpose = False
    sh = img.shape
    if direction == 'vertical':
        transpose = True
        if mask is not None:
            mask = mask.T
    brightness = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    if transpose:
        brightness = brightness.T
    energy = get_energy(brightness, mask)
    if method == 'shrink':
        ans = shrink(energy)
    else:
        ans = expand(energy)
    if transpose:
        return [i.T for i in ans]
    return ans
