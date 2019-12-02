from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Dense, BatchNormalization, Flatten, Dropout, Activation
from tensorflow.keras import Sequential
from os.path import join
from os import listdir
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from tensorflow.keras.models import load_model
from skimage.io import imread
from scipy.stats import bernoulli
from skimage.color import gray2rgb
from tensorflow.keras.optimizers import Adam

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import numpy as np
import tensorflow as tf


def read_img(img_dir, values):
    img = imread(img_dir)
    if len(img.shape) != 3:
        img = gray2rgb(img)
    sh = img.shape
    if values is not None:
        values[::2] /= sh[1]
        values[1::2] /= sh[0]
        values *= 100
        if values[1] > values[25] and values[5] > values[25]:
            img = img[::-1]
            values[1::2] = 99 - values[1::2]
    return img


def change_points(new_points, wc):
    new_points[wc, ::2] = 99 - new_points[wc, ::2]
    new_points[wc, :8:2] = new_points[wc, 6::-2]
    new_points[wc, 1:8:2] = new_points[wc, 7::-2]
    new_points[wc, 8:20:2] = new_points[wc, 18:7:-2]
    new_points[wc, 9:20:2] = new_points[wc, 19:8:-2]
    new_points[wc, 22:27:4] = new_points[wc, 26:21:-4]
    new_points[wc, 23:28:4] = new_points[wc, 27:22:-4]


def mirror_img(img, points):
    new_img = img.copy()
    which_change = np.argwhere(bernoulli.rvs(p=1, size=img.shape[0])).flatten()
    new_img[which_change] = new_img[which_change, :, ::-1, :]
    new_points = points.copy()
    change_points(new_points, which_change)
    return new_img, new_points

def kps_to_array(kps):
    ans = np.zeros((28,))
    for i in range(14):
        ans[i * 2] = kps.keypoints[i].x
        ans[i * 2 + 1] = kps.keypoints[i].y
    return ans


def rotate_img(img, points):
    kps = KeypointsOnImage([Keypoint(x=points[i], y=points[i + 1]) for i in range(0, len(points), 2)], shape=(100, 100, 3))
    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-25, 25),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        ),
    ]
    )
    image_aug, kps_aug = seq(image=img, keypoints=kps)
    return image_aug, kps_to_array(kps_aug)

def rotate_all_img(imgs, points):
    new_imgs = imgs.copy()
    new_points = points.copy()
    for i in range(len(imgs)):
        if np.random.rand() > 0.2:
            new_imgs[i], new_points[i] = rotate_img(new_imgs[i], new_points[i])
    return new_imgs, new_points


def generate_train_dataset(train_gt, train_img_dir, new_shape=(100, 100, 3)):
    dataset = []
    labels = []
    for key, values in train_gt.items():
        if key not in listdir(train_img_dir):
            print(key, "skip")
            continue
        img = read_img(join(train_img_dir, key), values)
        labels.append(values)
        dataset.append(resize(img, new_shape))
    return np.array(dataset), np.array(labels)


class DG():
    datagen = ImageDataGenerator()

def preprocess_dataset(dataset):
    dataset -= 0.4448004816970303
    dataset /= 0.2714417487523733


def train_detector(train_gt, train_img_dir, test_img_dir, fast_train=False, batch_size=50, epochs=11):
    if fast_train:
        epochs = 1
        return

    test_dataset, test_labels = generate_train_dataset(train_gt, test_img_dir)
    dataset, labels = generate_train_dataset(train_gt, train_img_dir)
    preprocess_dataset(dataset)
    preprocess_dataset(test_dataset)
    DG.datagen.fit(dataset)
    print(dataset.shape, labels.shape)
    model = Sequential([
        Conv2D(64, (3, 3), strides=(2, 2), input_shape=(100, 100, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        SpatialDropout2D(0.05),
        Conv2D(128, (3, 3), strides=(2, 2)),
        BatchNormalization(),
        Activation('relu'),
        SpatialDropout2D(0.05),
        Conv2D(256, (3, 3), strides=(2, 2)),
        BatchNormalization(),
        Activation('relu'),
        SpatialDropout2D(0.05),
        Flatten(),
        BatchNormalization(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.05),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.05),
        Dense(28),
    ])
    model = load_model('.\\facepoints_model.hdf5')
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(learning_rate=0.0003),
                  metrics=["mean_squared_error"])


    tensorboard = ModelCheckpoint('.\\facepoints_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    cur_data, cur_labels = mirror_img(dataset, labels)
    data = np.concatenate((dataset, cur_data), axis=0)
    labels = np.concatenate((labels, cur_labels), axis=0)
    cur_data, cur_labels = rotate_all_img(data, labels)
    print(cur_data.shape, cur_labels.shape)
    history = model.fit_generator(DG.datagen.flow(cur_data, cur_labels, batch_size=batch_size, shuffle=True),
                                  validation_data=DG.datagen.flow(test_dataset, test_labels, batch_size=batch_size, shuffle=True),
                                  validation_steps=len(test_dataset) / batch_size,
                                  steps_per_epoch=len(cur_data) / batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  #callbacks=[tensorboard],
                                  )
    model.save('.\\facepoints_model.hdf5')


def generate_answer(predict, test_img_dir, dic, ans, files):
    for i, file in enumerate(files):
        ans[file] = np.array(list(predict[i]))
        ans[file][::2] *= dic[file][0]
        ans[file][1::2] *= dic[file][1]


def detect(model, test_img_dir, batch_size=8):
    new_shape = (100, 100, 3)
    batch = []
    ans = {}
    coef_dic = {}
    files = []
    i = 0
    for file in listdir(test_img_dir):
        if i == batch_size:
            i = 0
            batch = np.array(batch)
            preprocess_dataset(batch)
            predict = model.predict(batch)
            generate_answer(predict, test_img_dir, coef_dic, ans, files)
            batch = []
            coef_dic = {}
            files = []
        img_dir = join(test_img_dir, file)
        img = read_img(img_dir, None)
        files.append(file)
        sh = img.shape
        coef_dic[file] = [sh[1] / 100, sh[0] / 100]
        batch.append(resize(img, new_shape))
        i += 1
    if i != 0:
        i = 0
        batch = np.array(batch)
        preprocess_dataset(batch)
        predict = model.predict(batch)
        generate_answer(predict, test_img_dir, coef_dic, ans, files)
        batch = []
        coef_dic = {}
    return ans
