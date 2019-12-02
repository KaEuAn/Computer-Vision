import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import block_reduce


def get_min_par(img_1, img_2, rr1=range(-15, 16), rr2=range(-15, 16)):
    # min_mse, min_dx1, min dy1
    mi = [np.inf, -1, -1]
    sh = img_1.shape
    for dx in rr1:
        for dy in rr2:
            cimg_1 = img_1[max(0, dx):min(sh[0], sh[0] + dx), max(0, dy):min(sh[1], sh[1] + dy)]
            cimg_2 = img_2[max(0, -dx):min(sh[0], sh[0] - dx), max(0, -dy):min(sh[1], sh[1] - dy)]
            xxx = mse(cimg_1, cimg_2)
            #print(xxx, dx, dy)
            if xxx < mi[0]:
                mi = [xxx, dx, dy]
                #print(mi, xxx)
    return mi


def pyramid(img_1, img_2):
    if img_1.shape[0] < 500 and img_2.shape[1] < 500:
        mi = get_min_par(img_1, img_2)
        print("mi", mi)
        return mi

    # сжимаем изображение в два раза
    new_img_1 = block_reduce(img_1, block_size=(2, 2), func=np.mean)
    new_img_2 = block_reduce(img_2, block_size=(2, 2), func=np.mean)

    sh = img_1.shape
    mi = pyramid(new_img_1, new_img_2)
    new_mi = [np.inf, mi[1], mi[2]]
    for ddx in (-1, 2):
        for ddy in (-1, 2):
            dx = ddx + mi[1] * 2
            dy = ddy + mi[2] * 2
            cimg_1 = img_1[max(0, dx):min(sh[0], sh[0] + dx), max(0, dy):min(sh[1], sh[1] + dy)]
            cimg_2 = img_2[max(0, -dx):min(sh[0], sh[0] - dx), max(0, -dy):min(sh[1], sh[1] - dy)]
            xxx = mse(cimg_1, cimg_2)
            if xxx < new_mi[0]:
                new_mi = [xxx, dx, dy]
    return new_mi


def align(img, g_coord):
    sh = img.shape
    print(img.shape, g_coord)
    h = img.shape[0] // 3
    blue_img = img[:h, :]
    green_img = img[h: h * 2, :]
    red_img = img[h * 2: h * 3, :]
    ans = [0, 0, 0]
    sh = red_img.shape
    red_ans = pyramid(green_img[int(sh[0] * 0.05):int(sh[0] * 0.95), int(sh[1] * 0.05):int(sh[1] * 0.95)],
                      red_img[int(sh[0] * 0.05):int(sh[0] * 0.95), int(sh[1] * 0.05):int(sh[1] * 0.95)])
    blue_ans = pyramid(green_img[int(sh[0] * 0.05):int(sh[0] * 0.95), int(sh[1] * 0.05):int(sh[1] * 0.95)],
                      blue_img[int(sh[0] * 0.05):int(sh[0] * 0.95), int(sh[1] * 0.05):int(sh[1] * 0.95)])
    ans[1] = (g_coord[0] - blue_ans[1] - h, g_coord[1] - blue_ans[2])
    ans[2] = (g_coord[0] - red_ans[1] + h, g_coord[1] - red_ans[2])
    ans[0] = np.zeros((h, red_img.shape[1]))
    print("red", ans[1])
    print("blue", ans[2])
    return ans
