from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras import Sequential
from os.path import join
from os import listdir
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.color import gray2rgb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np


def read_img(img_dir):
    img = load_img(img_dir, target_size=(299, 299))
    img = img_to_array(img)
    return img


def generate_train_dataset(train_gt, train_img_dir, new_shape=(299, 299, 3)):
    dataset = []
    labels = []
    for key, value in train_gt.items():
        if key[-5] == '0' and key[-6] == '0':
          print(key)
        if key not in listdir(train_img_dir):
            print(key, "skip")
            continue
        img = read_img(join(train_img_dir, key))
        if img.shape[-1] != 3:
            continue

        labels.append(value)
        dataset.append(preprocess_input(img))
    return np.array(dataset), to_categorical(np.array(labels), num_classes=50)


class DG():
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=(0.8, 1.1),
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=30,
        zoom_range=0.09,
    )


def preprocess_dataset(dataset):
    dataset -= 0.48815942497194714
    dataset /= 0.24515676705819864

def train_classifier(train_gt, train_img_dir, fast_train=False, epochs=11, batch_size=16, d=None):
    if fast_train:
        return
    if d is None:
      dataset, labels = generate_train_dataset(train_gt, train_img_dir)
      x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.15, random_state=41, stratify=labels)
      print(dataset.shape, labels.shape)
      print(np.mean(dataset), np.std(dataset))
    else:
      x_train, x_test, y_train, y_test = d

    old_model = Xception(include_top=False, weights='imagenet')
    p1 = GlobalAveragePooling2D()(old_model.output)
    d1 = Dense(256)(p1)
    b1 = BatchNormalization()(d1)
    a1 = Activation('relu')(b1)
    d2 = Dense(50, activation="softmax")(a1)

    new_model = Model(old_model.input, d2)
    for l in old_model.layers:#[:-6]:
        l.trainable = False
    #new_model = load_model(join(code_dir, 'birds_model.hdf5'))
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.0003),
                      metrics=["accuracy"])
    tensorboard = ModelCheckpoint(join(code_dir, 'birds_model.hdf5'), save_best_only=True, monitor='val_loss', mode='min')
    history = new_model.fit_generator(
        DG.datagen.flow(x=x_train, y=y_train, shuffle=True, batch_size=batch_size),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train)/batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard], )

    #new_model.save('.\\birds_model.hdf5')
    #new_model.save_weights('.\\birds_model_only_weights.hdf5')



def classify(model, test_img_dir, batch_size = 10, new_shape=(299, 299, 3)):
    batch = []
    ans = {}
    files = []
    i = 0
    for file in listdir(test_img_dir):
        if i == batch_size:
            i = 0
            batch = np.array(batch)
            predict = model.predict(batch)
            for i, filee in enumerate(files):
                ans[filee] = np.argmax(predict[i])
            batch = []
            files = []

        i += 1
        img_dir = join(test_img_dir, file)
        img = read_img(img_dir)
        batch.append(preprocess_input(img))
        files.append(file)
    if i != 0:
        i = 0
        batch = np.array(batch)
        predict = model.predict(batch)
        for i, filee in enumerate(files):
            ans[filee] = np.argmax(predict[i])
        batch = []
        files = []
    return ans
