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

# plt.imshow(mnist2[0].reshape(28,28))
# plt.show()

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(mnist2[i].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()


# Use K-Means
kmeans = KMeans(20)
mu_digits = kmeans.fit(mnist2).cluster_centers_

plt.figure(figsize=(16,6))
for i in range(int(2*(mu_digits.shape[0]/2))): # loop over all means
    plt.subplot(2,mu_digits.shape[0]/2,i+1)
    plt.imshow(mu_digits[i].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()

# KESKEN





