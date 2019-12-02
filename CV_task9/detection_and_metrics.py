# ============================== 1 Classifier model ============================
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DG():
    datagen = ImageDataGenerator(
        horizontal_flip=True,
    )


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Dense, BatchNormalization, Flatten, \
        Dropout, Activation, InputLayer
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        InputLayer(input_shape=(40, 100, 1)),
        Conv2D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.05),
        Dense(2),
        Activation('softmax')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0003),
                  metrics=["accuracy"])
    return model


def fit_cls_model(X, y, batch_size=4, epochs=100):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

    model = get_cls_model((40, 100, 1))
    #tensorboard = ModelCheckpoint('classifier_model.h5', save_best_only=True, monitor='loss',
    #                              mode='min')
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    #model.save('classifier_model.h5')
    return model


# =========================== 2 Classifier -> FCN =============================
def get_detection_model(model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Dense, BatchNormalization, Flatten, \
        Dropout, Activation, InputLayer
    from tensorflow.keras.optimizers import Adam

    new_model = Sequential()
    input_layer = InputLayer(input_shape=(None, None, 1))
    new_model.add(input_layer)
    for layer in model.layers:
        if "Flatten" in str(layer):
            flattened = True
            f_dim = layer.input_shape
            continue
        elif "Dense" in str(layer):
            input_shape = layer.input_shape
            output_dim = layer.get_weights()[1].shape[0]
            W, b = layer.get_weights()
            if flattened:
                shape = (f_dim[1], f_dim[2], f_dim[3], output_dim)
                new_W = W.reshape(shape)
                new_layer = Conv2D(output_dim,
                                   (f_dim[1], f_dim[2]),
                                   activation=layer.activation,
                                   padding='valid',
                                   weights=[new_W, b])
                flattened = False
            else:
                shape = (1, 1, input_shape[1], output_dim)
                new_W = W.reshape(shape)
                new_layer = Conv2D(output_dim,
                                   (1, 1),
                                   activation=layer.activation,
                                   padding="valid",
                                   weights=[new_W, b])
        else:
            new_layer = layer
        new_model.add(new_layer)

    new_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    return new_model


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    dest_sh = (220, 370)
    ans = {}
    for key, value in dictionary_of_images.items():
        sh = value.shape
        value = np.pad(value, ((0, dest_sh[0] - sh[0]), (0, dest_sh[1] - sh[1])))
        ans[key] = []
        pred = detection_model.predict(np.reshape(value, (1, 220, 370, 1)))[0]
        for i in range(sh[0] - 39):
            for j in range(sh[1] - 99):
                if pred[i][j][1] > 0.66:
                    ans[key].append([i, j, 40, 100, pred[i][j][1]])
    return ans


# =============================== 5 IoU ========================================
def bbox_to_standart(bbox):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]


def check(a):
    return max(a, 0)


def square(bbox):
    return check(bbox[2] - bbox[0]) * check(bbox[3] - bbox[1])


def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    first_bbox = first_bbox.copy()
    second_bbox = second_bbox.copy()
    bbox_to_standart(first_bbox)
    bbox_to_standart(second_bbox)

    bbox = [max(first_bbox[0], second_bbox[0]), max(first_bbox[1], second_bbox[1]), min(first_bbox[2], second_bbox[2]),
            min(first_bbox[3], second_bbox[3])]
    intersection = square(bbox)
    return intersection / (square(first_bbox) + square(second_bbox) - intersection)
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    iou_thr = 0.5
    tp = []
    tpfp = []
    all_count = 0
    for filename in gt_bboxes.keys():
        pred = sorted(pred_bboxes[filename], key=lambda x: x[4], reverse=True)
        gt = set()
        for i in gt_bboxes[filename]:
            gt.add(tuple(i))
        all_count += len(gt)
        for pimg in pred:
            thr = 0
            razm = []
            for gimg in gt:
                tmp = calc_iou(pimg, list(gimg))
                if tmp >= thr:
                    thr = tmp
                    razm = gimg
            if thr > iou_thr:
                tp.append(pimg[4])
                tpfp.append(pimg[4])
                gt.remove(razm)
            else:
                tpfp.append(pimg[4])
    tpfp = sorted(tpfp, reverse=True)
    tp = sorted(tp, reverse=True)
    pos = 0
    ans = [[0, 1, 0]]
    for i, c in enumerate(tpfp):
        if i < len(tpfp) - 1 and tpfp[i] == tpfp[i + 1]:
            continue
        while pos != len(tp) and tp[pos] >= c:
            pos += 1
        ans.append((pos / all_count, pos / (i + 1), c))

    ans.append((len(tp) / all_count, len(tp) / len(tpfp), 0))
    res = 0
    for i in range(len(ans) - 1):
        res += (ans[i + 1][0] - ans[i][0]) * (ans[i + 1][1] + ans[i][1])
    res /= 2
    return res


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.25):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    ans = {}
    for key, value in detections_dictionary.items():
        ans[key] = []
        value = sorted(value, key=lambda x: x[4], reverse=True)
        for detection in value:
            for check in ans[key]:
                if calc_iou(check, detection) >= iou_thr:
                    break
            else:
                ans[key].append(detection)
    return ans
