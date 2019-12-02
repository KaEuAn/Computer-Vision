import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.transform import resize
from skimage.color import rgb2gray
from numpy.linalg import inv
from math import sqrt
from numpy.linalg import norm
from numpy.linalg import norm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.ndimage import sobel
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def fit_and_classify():
    pass


class Solution:

    def __init__(self, img):
        self.img = img
        self.brightness = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        self.sh = self.brightness.shape
        self.eps = 1e-9

    def get_bin_number(self, value, bin_count=8):
        ans = (value + np.pi) // (2 * np.pi / bin_count)
        if ans >= bin_count:
            return bin_count - 1
        return int(ans)

    def get_sobel(self):
        return sobel(self.brightness, axis=0, mode='nearest'), sobel(self.brightness, axis=1, mode='nearest')

    def calculate(self):
        grad_x, grad_y = self.get_sobel()
        self.grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        self.direction_grad = np.arctan2(grad_y, grad_x)

    def get_all_bins(self, bin_count=8, cell_rows=4, cell_cols=4):
        all_bins = []
        for i in range(0, self.sh[0], cell_rows):
            all_bins.append([])
            for j in range(0, self.sh[1], cell_cols):
                bins = [0] * bin_count
                for x in range(i, min(self.sh[0], i + cell_rows)):
                    for y in range(j, min(self.sh[1], j + cell_cols)):
                        bins[self.get_bin_number(self.direction_grad[x][y])] += 1
                all_bins[-1].append(bins)
        self.all_bins = np.array(all_bins)

    def get_feature(self, block_row_cells=3, block_col_cells=3):
        shape = self.all_bins.shape
        features = np.array([])
        for i in range(shape[0] - block_row_cells):
            for j in range(shape[1] - block_col_cells):
                feature = np.resize(self.all_bins[i: i + block_row_cells, j: j + block_col_cells],
                                    (block_col_cells * block_row_cells,))
                feature = feature / np.sqrt(norm(feature) + self.eps)
                features = np.concatenate((features, feature), axis=None)
        self.features = features

    def run(self):
        self.calculate()
        self.get_all_bins()
        self.get_feature()
        return self.features


def extract_hog(img, bin_count=8, cell_rows=4, cell_cols=4):
    sh = img.shape
    new_img = resize(img, (60, 60), anti_aliasing=True)
    solution = Solution(new_img)
    return solution.run()


def fit_and_classify(train_features, train_labels, test_features):
    model = LinearSVC(C=0.009, max_iter=2500)
    model.fit(train_features, train_labels)
    return model.predict(test_features)
