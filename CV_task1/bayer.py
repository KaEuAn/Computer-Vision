import numpy as np
from skimage import img_as_float32
from math import floor, log10


def get_bayer_masks(n_rows, n_cols):
    r = np.array([[0, 1], [0, 0]]);
    g = np.array([[1, 0], [0, 1]]);
    b = np.array([[0, 0], [1, 0]])
    rr = np.tile(r, (n_rows, n_cols))[:n_rows, :n_cols]
    gg = np.tile(g, (n_rows, n_cols))[:n_rows, :n_cols]
    bb = np.tile(b, (n_rows, n_cols))[:n_rows, :n_cols]
    matrix = np.dstack((rr, gg, bb))
    return matrix


def get_colored_img(raw_img):
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    masks[:, :, 0] *= raw_img
    masks[:, :, 1] *= raw_img
    masks[:, :, 2] *= raw_img
    return masks


def bilinear_interpolation(colored_img):
    colored_img = colored_img
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    sh = colored_img.shape
    copy_img = np.zeros_like(colored_img)
    for i in range(1, sh[0] - 1):
        for j in range(1, sh[1] - 1):
            for k in range(sh[2]):
                if mask[i, j, k] == 0:
                    copy_img[i][j][k] = np.sum(colored_img[i - 1: i + 2, j - 1: j + 2, k]) / np.sum(
                        mask[i - 1: i + 2, j - 1: j + 2, k])
                else:
                    copy_img[i][j][k] = colored_img[i][j][k]
    return copy_img


# row_diff, col_diff, r/g/b, koef
filter_list = [
    [  # G at R filter
        (0, 0, 0, 4),
        (-2, 0, 0, -1),
        (2, 0, 0, -1),
        (0, -2, 0, -1),
        (0, 2, 0, -1),
        (1, 0, 1, 2),
        (-1, 0, 1, 2),
        (0, 1, 1, 2),
        (0, -1, 1, 2),
    ],
    [  # G at B filter
        (0, 0, 2, 4),
        (-2, 0, 2, -1),
        (2, 0, 2, -1),
        (0, -2, 2, -1),
        (0, 2, 2, -1),
        (1, 0, 1, 2),
        (-1, 0, 1, 2),
        (0, 1, 1, 2),
        (0, -1, 1, 2),
    ],
    [  # R at G R row, B col
        (0, 0, 1, 5),
        (-1, -1, 1, -1),
        (1, -1, 1, -1),
        (1, 1, 1, -1),
        (-1, 1, 1, -1),
        (-2, 0, 1, 0.5),
        (2, 0, 1, 0.5),
        (0, 2, 1, -1),
        (0, -2, 1, -1),
        (0, 1, 0, 4),
        (0, -1, 0, 4),
    ],
    [  # R at G R col, B row
        (0, 0, 1, 5),
        (-1, -1, 1, -1),
        (1, -1, 1, -1),
        (1, 1, 1, -1),
        (-1, 1, 1, -1),
        (-2, 0, 1, -1),
        (2, 0, 1, -1),
        (0, 2, 1, 0.5),
        (0, -2, 1, 0.5),
        (1, 0, 0, 4),
        (-1, 0, 0, 4),
    ],
    [  # R at B col, B row
        (0, 0, 2, 6),
        (-1, -1, 0, 2),
        (1, -1, 0, 2),
        (1, 1, 0, 2),
        (-1, 1, 0, 2),
        (-2, 0, 2, -1.5),
        (2, 0, 2, -1.5),
        (0, 2, 2, -1.5),
        (0, -2, 2, -1.5),
    ],
    [  # B at G B row, R col
        (0, 0, 1, 5),
        (-1, -1, 1, -1),
        (1, -1, 1, -1),
        (1, 1, 1, -1),
        (-1, 1, 1, -1),
        (-2, 0, 1, 0.5),
        (2, 0, 1, 0.5),
        (0, 2, 1, -1),
        (0, -2, 1, -1),
        (0, 1, 2, 4),
        (0, -1, 2, 4),
    ],
    [  # B at G R col, B row
        (0, 0, 1, 5),
        (-1, -1, 1, -1),
        (1, -1, 1, -1),
        (1, 1, 1, -1),
        (-1, 1, 1, -1),
        (-2, 0, 1, -1),
        (2, 0, 1, -1),
        (0, 2, 1, 0.5),
        (0, -2, 1, 0.5),
        (1, 0, 2, 4),
        (-1, 0, 2, 4),
    ],
    [  # B at R col, R row
        (0, 0, 0, 6),
        (-1, -1, 2, 2),
        (1, -1, 2, 2),
        (1, 1, 2, 2),
        (-1, 1, 2, 2),
        (-2, 0, 0, -1.5),
        (2, 0, 0, -1.5),
        (0, 2, 0, -1.5),
        (0, -2, 0, -1.5),
    ],

]


def improved_interpolation(raw_img):
    colored_img = get_colored_img(raw_img)
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    sh = colored_img.shape
    copy_img = np.zeros_like(colored_img, dtype='float')
    for i in range(2, sh[0] - 2):
        for j in range(2, sh[1] - 2):
            for k in range(sh[2]):
                if mask[i, j, k] == 0:
                    if k == 1 and mask[i, j, 0] != 0:
                        cur_filter = filter_list[0]
                    elif k == 1:
                        cur_filter = filter_list[1]
                    elif k == 0 and mask[i, j - 1, 0] != 0:
                        cur_filter = filter_list[2]
                    elif k == 0 and mask[i - 1, j, k] != 0:
                        cur_filter = filter_list[3]
                    elif k == 0:
                        cur_filter = filter_list[4]
                    elif k == 2 and mask[i, j - 1, 2] != 0:
                        cur_filter = filter_list[5]
                    elif k == 2 and mask[i - 1, j, 2] != 0:
                        cur_filter = filter_list[6]
                    else:
                        cur_filter = filter_list[7]
                    for id, jd, kd, alpha in cur_filter:
                        copy_img[i, j, k] += alpha * colored_img[i + id, j + jd, kd] / 8
                else:
                    copy_img[i][j][k] = colored_img[i][j][k]
    return np.clip(copy_img, 0, 255).astype("uint8")


def mse(a, b):
    sh = a.shape
    return np.sum((a - b)**2)/sh[0] / sh[1]

def compute_psnr(img_pred, img_gt):
    mse_value = mse(img_pred, img_gt)
    if mse_value < 1e-14:
        raise ValueError()
    return 10 * log10((np.max(img_gt)**2) / mse_value)