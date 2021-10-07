import random
import numpy as np
from numpy import where
from matplotlib import pyplot
from collections import Counter
from sklearn.datasets import make_classification

def find_knn(minorDataset, X, k):
	distList = np.array([0], dtype=float)
	sortedKnnList = np.array([[X]])
	for Xnn in minorDataset:
		if ( np.array_equal(Xnn, X) ):
			continue
		diffX = Xnn[0] - X[0]
		diffY = Xnn[1] - X[1]
		dist = np.sqrt( np.square(diffX) + np.square(diffY) )
		i = 0 if len(distList) == 0 else np.where(distList < dist)[0][0]
		distList = np.insert(distList, i, dist)
		sortedKnnList = np.insert(sortedKnnList, i, Xnn, axis=1)

	knnList = sortedKnnList[0][-1-k:-1]
	return knnList

def make_random_index(start, end, indexList):
	index = random.randint(start, end)
	while index in indexList:
		index = random.randint(0, end)
	indexList.append(index)
	return index

def SMOTE(minorDataset, k, count):
	if (k >= 2):
		xIndexList = []
		syntheticDataset = []
		for n in range(0, count):
			xIndex = make_random_index(0, len(minorDataset) - 1, xIndexList)
			X = minorDataset[xIndex]
			knnList = find_knn(minorDataset, X, k)
			knnIndex = random.randint(0, k - 1)
			Xnn = knnList[knnIndex]
			u = random.uniform(0, 1)
			syntheticData = X + u * (Xnn - X)
			syntheticDataset.append(syntheticData)
			print('X: {0}, Xnn: {1}, u: {2}'.format(X, Xnn, u))

		return np.array(syntheticDataset)

# define dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=1, n_redundant=0,
	n_clusters_per_class=1, weights=[0.90], random_state=3)

counter = Counter(y)
print(counter)

for label, unused in counter.items():
	rowIndex = where(y == label)[0]
	color = 'g' if label == 1 else 'darkgrey'
	classLabel = 'Minor' if label == 1 else 'Major'
	pyplot.scatter(X[rowIndex, 0], X[rowIndex, 1], c = color, label=classLabel)

minorIndex = where(y == 1)[0]
minorDataset = X[minorIndex]

# dataset for debugging
#minorDataset = np.array([[1, 1], [3, 5], [4, 4], [5, 3], [7, 7], [8, 8]])

# SMOTE
syntheticDataset = SMOTE(minorDataset, k = 3, count = len(minorDataset))

pyplot.scatter(syntheticDataset[:, 0], syntheticDataset[:, 1], c = 'r', label='Synthetic')
pyplot.title("SMOTE", fontsize=15)
pyplot.legend()
pyplot.show()

