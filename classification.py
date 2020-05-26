import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

data_num = 100
data_dim = 2
num_classes = 3

data1 = [0.5, -0.5] + 0.15 * np.random.randn(data_num, data_dim)
data2 = -0.5 + 0.2 * np.random.randn(data_num, data_dim)
data3 = 0.35 + 0.25 * np.random.randn(data_num, data_dim)

data = np.concatenate((data1, data2, data3), axis=0)
all_num = data_num * num_classes

labels = np.zeros(all_num)
labels[data_num:2 * data_num] = 1
labels[-data_num:] = 2
colors = ["C%d" % l for l in labels]

# one hot
y_train = keras.utils.to_categorical(labels, num_classes=3)
print(y_train.shape)

plt.scatter(*data.T, alpha=0.5, c=colors)
plt.grid()
plt.show()

model = Sequential()
model.add(Dense(7, input_shape=(2,), activation="tanh"))
model.add(Dense(7, activation="tanh"))
model.add(Dense(7, activation="tanh"))
model.add(Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history = model.fit(data, y_train,
                    batch_size=1,
                    epochs=10)
