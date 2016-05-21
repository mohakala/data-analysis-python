from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os

"""

K-Means 
Unsupervised learning of images (hand-written digits)
Following the example from Python Notebook:
http://nbviewer.jupyter.org/github/temporaer/tutorial_ml_gkbionics/tree/master/

"""


# Set the size of the training set and the number of clusters 
nTrainingSet=10000
nClusters=20


# Fetch data
if (not os.path.exists("/Python34/test_mnist/mldata")):
    mnist = fetch_mldata('MNIST original', data_home='test_mnist')
else:
    mnist = fetch_mldata('MNIST original')
print('data,target shapes:',mnist.data.shape,mnist.target.shape)
print('unique targets:',np.unique(mnist.target))


# Select the training set
mnist.data, mnist.target = shuffle(mnist.data,mnist.target)
mnist2=mnist.data[-nTrainingSet:]
label2=mnist.target[-nTrainingSet:]
print(type(mnist2))
print(mnist2.shape)


# Plot
plt.rc("image", cmap="binary")

# plt.imshow(mnist2[0].reshape(28,28))
# plt.show()

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(mnist2[i].reshape(28,28))
    plt.title(label2[i])
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.suptitle("Examples from the dataset", size=16)
plt.show()



# Use K-Means
kmeans = KMeans(nClusters)
mu=kmeans.fit(mnist2)
mu_digits=mu.cluster_centers_
mu_lab=mu.labels_
print('mu_digits.shape:',mu_digits.shape)




# Choose and show data to be predicted
sampleIndex=int(nTrainingSet/2) # Take smpl from middle ofset
sample=mnist2[sampleIndex]
plt.imshow(sample.reshape(28,28))
plt.title('Sample to be predicted:'+str(label2[sampleIndex]))
plt.show()
print('Sample to be predicted: '+str(label2[sampleIndex]))


# Prediction based on clusters
prediction=kmeans.predict(sample)
print('Prediction: Cluster label = ', prediction) 


# Plot examples belonging to cluster 'prediction'
cluster=prediction
i=0
for ii in range(400):
    if (mu_lab[ii]==cluster):
        plt.subplot(2,6,i+1)
        plt.imshow(mnist2[ii].reshape(28,28))
        plt.xticks(())
        plt.yticks(())
        i+=1
        if (i>10):
            break
plt.tight_layout()
plt.suptitle("The sample belongs to cluster "+str(prediction)+", example cases:", size=16)
plt.show()


# Show clusters and the sample to be predicted
print('Plotting the clusters that were found')
plt.figure(figsize=(16,6))
for i in range(int(2*(mu_digits.shape[0]/2))): # loop over all means
    plt.subplot(2,mu_digits.shape[0]/2+1,i+1)
    plt.imshow(mu_digits[i].reshape(28,28))
    plt.title(i)
    plt.xticks(())
    plt.yticks(())
plt.subplot(2,mu_digits.shape[0]/2+1,i+2)
plt.imshow(sample.reshape(28,28))
plt.title('Predicted label:'+str(prediction))
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.suptitle("The clusters that were found", size=16)
plt.show()









