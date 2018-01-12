import keras
import skimage.io
import numpy as np

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


test_real_data()
