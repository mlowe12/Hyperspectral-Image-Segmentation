
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.misc import comb

## imports data shd reshapes it so that it can be worked with.
hyperSpectralImages = spio.loadmat('Indian_pines_corrected.mat')
imageGroundTruth = spio.loadmat('Indian_pines_gt.mat')

imageArray = hyperSpectralImages['indian_pines_corrected'] # imageArray is a 145*145*200 tensor. The third dimension stands for the # of bands
groundTruthArray = imageGroundTruth['indian_pines_gt']

featureArray = np.zeros((21025, 200)) # 'stretch' the 145*145*200 tensor to an 21025*200 array, which has 21025 data points and 200 features
pixelLabel = groundTruthArray.reshape(21025)

for i in range(50,199):
	featureArray[:, i] = imageArray[:,:,i].reshape(21025)
featureNorm = scale(featureArray)


# Random index scoring
def rand_index_score(clusters, classes):

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

## Radial Neigbor biasing to make the clusters more "fuzzy"
def NeighborBias(sqrmap,maxClasses,radius):

	xsize,ysize = sqrmap.shape

	# Evolution of map
	newMap = np.zeros((xsize,ysize))

	neighborsClasses = np.zeros(maxClasses)
	neighborsClasses.fill(-10)
	neighborsHist = np.zeros(maxClasses)

	for i in range(xsize):
		for j in range(ysize):

			neighborsHist.fill(0)

			for ii in range(i-radius,i+radius+1):
				for jj in range(j-radius,j+radius+1):

					if(ii<0 or ii>=xsize or jj<0 or jj>=ysize):
						continue

					for k in range(maxClasses):
						if(neighborsClasses[k] == sqrmap[ii][jj]):
							neighborsHist[k]+=1
							break
						elif(neighborsClasses[k] == -10):
							neighborsClasses[k] = sqrmap[ii][jj]
							neighborsHist[k]+=1
							break

			newMap[i][j] = neighborsClasses[neighborsHist.argmax()]


	return newMap



# PCA reduces the dimentions of the data from 200 to 20
pca = PCA(n_components=20)
newFeature = pca.fit_transform(featureNorm)
# Preforms the k-means on the transformed data set
kmeans = KMeans(n_clusters = 15, max_iter = 1000).fit(newFeature)
result =kmeans.labels_.reshape((145,145))

## removes everything without a label
newmap = NeighborBias(result,16,9)
newmap =newmap.astype(np.int64)
a = pixelLabel[np.where(pixelLabel != 0)]
b = newmap.reshape(21025)[np.where(pixelLabel!=0)]
c = newmap.reshape(21025)
for i in range(21025):
    if pixelLabel[i]==0:
        c[i]=-1

print ('Rand index score: ')
print(rand_index_score(a, b))
print ('Adjusted rand index score: ')
print(metrics.adjusted_rand_score(a ,b))

## Plot the final results
plt.figure()
plt.imshow(c.reshape(145,145))
plt.show()






