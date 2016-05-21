from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os


# Fetch data

if (not os.path.exists("/Python34/test_mnist/mldata")):
    mnist = fetch_mldata('MNIST original', data_home='test_mnist')
else:
    mnist = fetch_mldata('MNIST original')
print('data,target shapes:',mnist.data.shape,mnist.target.shape)
print('unique targets:',np.unique(mnist.target))


# Select the training set

mnist.data, mnist.target = shuffle(mnist.data,mnist.target)
mnist2=mnist.data[-5000:]
print(type(mnist2))
print(mnist2.shape)


# Plot

plt.rc("image", cmap="binary")
plt.imshow(mnist2[0].reshape(28,28))
plt.show()
plt.imshow(mnist2[1].reshape(28,28))
plt.show()


