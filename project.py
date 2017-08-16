import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA 
from sklearn.manifold import Isomap 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA

hyperSpectralImages = spio.loadmat('Indian_pines_corrected.mat')
imageGroundTruth = spio.loadmat('Indian_pines_gt.mat')

imageArray = hyperSpectralImages['indian_pines_corrected'] # imageArray is a 145*145*200 tensor. The third dimension stands for the # of bands
groundTruthArray = imageGroundTruth['indian_pines_gt']

featureArray = np.zeros((21025, 200)) # 'stretch' the 145*145*200 tensor to an 21025*200 array, which has 21025 data points and 200 features
pixelLabel = groundTruthArray.reshape(21025)

for i in range(200):
	featureArray[:, i] = imageArray[:,:,i].reshape(21025)
# feature standardization
featureNorm = scale(featureArray)

# PCA
#kpca = KernelPCA(n_components=20)#kernel="rbf")
pca = PCA(n_components=20)
newFeature = pca.fit_transform(featureNorm)

kmeans = KMeans(n_clusters = 2, max_iter = 10000).fit(newFeature)

result = kmeans.labels_.reshape((145,145))

#plt.plot(result)
plt.figure()
plt.imshow(result)
#plt.imshow(groundTruthArray)
plt.show()