import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

SIZE = 10000
GROUPS = 20

mnist = tf.keras.datasets.mnist
_, digits = mnist.load_data()

X = digits[0]   # Image data of shape (10000, 28, 28)
y = digits[1]   # Labels of shape (10000, )

X = X.reshape(SIZE, 784)

# Randomly intialize a table with integer value in [0, 19]
# group[i] represents the group that i-th data point belongs to
group = np.random.randint(0, GROUPS, size=SIZE)

# Compute the mean (representative) of each group
representative = [np.zeros(784) for _ in range(GROUPS)]
for group_index in range(GROUPS):
    group_data = X[group == group_index, :]
    representative[group_index] = group_data.mean(axis=0)


# k-means algorithm
J = float('inf')  # Current cost
Jprev = 0  # Previous cost
first_iteration = True

while first_iteration or abs(J - Jprev) > 1e-5 * J:
    first_iteration = False
    Jprev = J
    sum = 0
    
    # Compute the mean (representative) of each group
    for group_index in range(GROUPS):
        group_data = X[group == group_index, :]
        representative[group_index] = group_data.mean(axis=0)

    for i in range(SIZE):
        distances = (np.linalg.norm(X[i] - representative, axis=1) ** 2) 
        sum += min(distances)
        min_j = np.argmin(distances)
        group[i] = min_j
    
    J = sum / SIZE

fig, ax = plt.subplots(2, 10, figsize=(8,3))
for axi, rep in zip(ax.flat, representative):
    axi.set(xticks=[], yticks=[]) # clear the ticks of each subplot
    axi.imshow(rep.reshape(28, 28))

plt.show()