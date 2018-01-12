import keras
import skimage.io
import numpy as np
import scipy.stats as stats

import pylab as plt

num_classes = 2
model_name = "../models/hand_classification/hand_classification_model.h5"
model = keras.models.load_model(model_name)


def load_data(path, use_truth):
    data = skimage.io.imread_collection(path)
    x_data = np.array([data[i] for i in range(len(data.files))])
    y_data = None
    if use_truth:
        y_data = np.array([0 if name[-5] == "L" else 1 for name in data.files])
        y_data = keras.utils.to_categorical(y_data, num_classes)

    return x_data, y_data


def test_real_data():
    x_data, y_data = load_data("./data/real/*.jpg", True)
    print(x_data.shape, y_data.shape)

    x_data = x_data.astype('float32')
    x_data /= 255

    scores = model.evaluate(x_data, y_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def predict_real_hand():
    x_data, _ = load_data("./data/real/*.jpg", False)
    x_data = x_data.astype('float32')
    x_data /= 255

    outputs = model.predict(x_data)
    print(outputs)
    return outputs


def predict_no_hand():
    x_data, _ = load_data("./data/no-hands/*.jpg", False)
    x_data = x_data.astype('float32')
    x_data /= 255

    outputs = model.predict(x_data)
    print(outputs)
    return outputs


def draw_max_distribution():
    n_data = 100
    n_bins = 10

    real_hands_scores = sorted(np.max(predict_real_hand(), axis=1)[:n_data])
    no_hands_scores = sorted(np.max(predict_no_hand(), axis=1)[:n_data])
    colors = ['red', 'tan']
    labels = ['with hand', 'without hand']
    data_to_draw = np.vstack((real_hands_scores, no_hands_scores))
    print(data_to_draw)
    plt.hist(data_to_draw.T, bins=n_bins, color=colors, label=labels)
    plt.legend(prop={'size': 10})
    plt.title("Max score of the  scores of two classes")
    plt.show()  # use may also need add this


def draw_difference_distribution():
    n_data = 100
    n_bins = 10

    real_hands_scores = sorted([np.abs(x[0] - x[1]) for x in predict_real_hand()[:n_data]])
    no_hands_scores = sorted([np.abs(x[0] - x[1]) for x in predict_no_hand()[:n_data]])
    colors = ['red', 'tan']
    labels = ['with hand', 'without hand']
    data_to_draw = np.vstack((real_hands_scores, no_hands_scores))
    print(data_to_draw)
    plt.hist(data_to_draw.T, bins=n_bins, color=colors, label=labels)
    plt.legend(prop={'size': 10})
    plt.title("Difference between scores of two classes")
    plt.show()  # use may also need add this


draw_difference_distribution()
# predict_real_hand()
