import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python import keras

data_num = 100
data_dim = 2

data1 = [0.5, -0.5] + 0.15 * np.random.randn(data_num, data_dim)
data2 = -0.5 + 0.2 * np.random.randn(data_num, data_dim)
data3 = 0.35 + 0.25 * np.random.randn(data_num, data_dim)

data = np.concatenate((data1, data2, data3), axis=0)
all_num = data_num * 3

labels = np.zeros(all_num)
labels[data_num:2 * data_num] = 1
labels[-data_num:] = 2
colors = ["C%d" % l for l in labels]

# one hot
y_train = keras.utils.to_categorical(labels, num_classes=3)
print(y_train)

plt.scatter(*data.T, alpha=0.5, c=colors)
plt.grid()
plt.show()
