import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(range(5))
ax = fig.add_subplot(323)
ax.plot(range(6))
ax = fig.add_subplot(324)
ax.plot(range(7))

# Second method
ax = fig.add_subplot(3, 2, (5, 6))
ax.plot(range(8))

plt.show()
