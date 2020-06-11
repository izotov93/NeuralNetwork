from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf

import os.path
from models import mnist_cnn_train, mnist_cnn_model, cnn_digits_predict


# загрузка данных MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# выборка по конкреной цифрр
#print('Введите цифру для выборки: ')
sel_num = input('Введите цифру для выборки: ')

try:
    sel_num = int(sel_num)
    if (sel_num < 10) and (sel_num >=  0):
        print('Цифра принята, ждите обработки данных')
    else:
        print('Число введено не верно, введите в диапозоне от 0 до 9')
        exit()
except Exception as e:
    print('Число введено не верно, введите в диапозоне от 0 до 9')
    exit()

#sel_num = 6
num = 1000
images = X_train[:num]
labels = Y_train[:num]


select_images = []
select_labels = []

for i in range(num):
    if labels[i] == sel_num:
        select_images.append(images[i])
        select_labels.append(labels[i])

count = np.sum(labels == sel_num)
num_row = math.ceil(math.sqrt(count))

# Вывод всех вхождений массив
fig, axes = plt.subplots(num_row, num_row,
                         figsize=(1.5*num_row,2*num_row))
for i in range(count):
    ax = axes[i//num_row, i%num_row]
    ax.imshow(select_images[i], cmap='gray_r')
    ax.set_title('Label: {}'.format(select_labels[i]))
plt.tight_layout()
plt.show()

# Среднее арифметическое значений элементов массива
result = np.mean(select_images, axis=0)
# убираем лишний шум при усреднении, все элементы массива, значения меньше 50 обнуляются.
low_values_flags = result < 30
result[low_values_flags] = 0

#Вывод итогового изображения
pixels = result.reshape((28, 28))
plt.imshow(pixels, cmap='gray_r')
plt.show()
plt.imsave('temp.png', pixels, cmap='gray_r')

# Нейросеть для анализа изображения
if os.path.exists('cnn_digits_28x28.h5'):
    model = tf.keras.models.load_model('cnn_digits_28x28.h5')
else:
    model = mnist_cnn_model()
    mnist_cnn_train(model)
    model.save('cnn_digits_28x28.h5')

result = cnn_digits_predict(model, 'temp.png')

print('Изображенная цифра - {}, Точность - {} %'.format(result[0], result[1][result[0]]))
