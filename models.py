'''
Модели нейронной сети

'''

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import keras

def mnist_cnn_model():
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   num_classes = 10  # Number of outputs
   model = Sequential()
   model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(image_size, image_size, num_channels)))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='relu'))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['accuracy'])
   return model


def mnist_cnn_train(model):
   (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()
   # Get image size
   image_size = 28
   num_channels = 1  # 1 for grayscale images

   train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
   train_data = train_data.astype('float32') / 255.0
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

   val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
   val_data = val_data.astype('float32') / 255.0
   val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

   model.fit(train_data, train_labels_cat, epochs=10, batch_size=64,
        validation_data=(val_data, val_labels_cat))
   return model


def cnn_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file,
            target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))

   result = []
   result.append(model.predict_classes([img_arr])[0])
   result.append(model.predict_proba([img_arr])[0])
   #result.append(number)
   #result.append(loss)

   return result