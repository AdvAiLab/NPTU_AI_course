import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

x_train = np.arange(0, 1, 0.01)
y_train = np.sin(x_train * (2 * np.pi))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(x_train, y_train, label='Ground Truth')
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")

batch_size = 1
epochs = 100

model = Sequential()
model.add(Dense(5, activation='tanh', input_shape=(1,)))
model.add(Dense(5, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

line = ax1.plot([], [], linestyle="-.", label="Predict")
plt.legend()

for ep in range(epochs):
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=1, verbose=1)
    y_predict = model.predict(x_train)
    line[0].set_data([x_train, y_predict])
    if ep > 0:
        ax2.plot([ep - 1, ep], [prev_loss, history.history["loss"][0]], c="C0")
    prev_loss = history.history["loss"][0]
    ax1.set_title("Epochs: %d" % ep)
    plt.pause(0.1)
    plt.draw()
plt.show()
