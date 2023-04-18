import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Plot your data on the original axes
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
ax.plot(x, y)

# Create a new axis object with no ticks or labels
circle_ax = fig.add_axes([0.5, 0.5, 0.1, 0.1], frameon=False, xticks=[], yticks=[])

# Draw a circle on the new axis object
circle_ax.add_artist(plt.Circle((0.5, 0.5), 0.1, color='red'))

plt.show()

