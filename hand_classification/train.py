from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import skimage.io
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


def load_data():
    train_data = skimage.io.imread_collection("./data/train/*.jpg")
    y_train = [0 if name[-5] == "L" else 1 for name in train_data.files]
    x_train = [train_data[i] for i in range(len(y_train))]

    test_data = skimage.io.imread_collection("./data/test/*.jpg")
    y_test = [0 if name[-5] == "L" else 1 for name in test_data.files]
    x_test = [test_data[i] for i in range(len(y_test))]

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


batch_size = 64
v_batch_size = 128

num_classes = 2
epochs = 1000
save_dir = os.path.join(os.getcwd(), 'data')
model_name = 'hand_classification_model.h5'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = load_data()

print(x_train[0].shape)


print("loading done..")
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(np.sum(y_train), len(y_train))
print(np.sum(y_test), len(y_test))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

import os.path

model_name = "./data/hand_classification_model.h5"
if os.path.isfile(model_name):
  print("loading model...")
  model = keras.models.load_model(model_name)

# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

checkpointer = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='auto', period=1)

train_image_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
validation_image_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

model.fit_generator(train_image_generator.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs, steps_per_epoch=len(x_train) // batch_size,
                    validation_data=validation_image_generator.flow(x_test, y_test, batch_size=v_batch_size),
                    validation_steps=len(x_test) // v_batch_size,
                    callbacks=[checkpointer])

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
