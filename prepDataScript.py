from fileinput import filename
import scipy.io
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import train_test_split


# define custom kernels
def linear_kernel(X, Y):
    return np.dot(X, Y.T)


def pairwise_distances(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    D = np.zeros([m, n])
    for jj in range(m):
        x = X[jj, :]
        for kk in range(n):
            y = Y[kk, :]
            D[jj, kk] = np.linalg.norm(x-y)**2
    return D

def gaussian_kernel_parameter(x, y, s):
    distmatrix = pairwise_distances(x, y)
    return np.exp(-distmatrix/(s**2))

def gaussian_kernel(s):
    return lambda X, Y : gaussian_kernel_parameter(X, Y, s)

# compute the percentage that are incorrect
def my_discrepancy(x, y):
    n_incorrect = np.count_nonzero(x - y)
    return n_incorrect / x.shape[0]


def laplace_kernel_parameter(x,y,s):
    distmatrix = pairwise_distances(x,y)
    return np.exp(-distmatrix/s)

def laplace_kernel(s):
    return lambda X, Y: laplace_kernel_parameter(X,Y,s)





# load data
fileName = 'Indian_pines_corrected'
labelsName = 'Indian_pines_gt'
data = scipy.io.loadmat(fileName)
data = data['indian_pines_corrected']

labels = scipy.io.loadmat(labelsName)
labels = labels['indian_pines_gt']

# get dimensions of the dataset
npoints = [data.shape[0], data.shape[1]]
nbands = data.shape[2]
classlabels = np.unique(labels)
nclasses = len(classlabels)

# split dataset into each class and store as dictionaries
spectra = {}
coordinates = {}

for k in range(nclasses):
    spectra[k] = []
    coordinates[k] = []

for ii in range(data.shape[0]):
    for jj in range(data.shape[1]):
        k = labels[ii, jj] #error here due to tuple 
        spectra[k].append(data[ii, jj, :])
        coordinates[k].append([ii, jj])

# split dataset into train and test
samplesperclass = []
Xtrain = []
Xtest = []
Ytrain = []
Ytest = []
traincoordinates = {}
testcoordinates = {}

# ignore class 0, which is unlabeled
#random.seed(10)
classes_to_classify = [1, 4, 5, 10]  # which categories to compare
per = 0.6  # percentage used to train data

for k in classes_to_classify:
    N = len(spectra[k])
    n = round(per*N)

    # (x,y) coordinates for the k-th class
    temp = coordinates[k]
    random.shuffle(temp)
    traincoordinates[k] = temp[:n]
    testcoordinates[k] = temp[n:]

    # put spectra into array
    for coor in traincoordinates[k]:
        Xtrain.append(data[coor[0], coor[1], :])
        Ytrain.append(k)

    for coor in testcoordinates[k]:
        Xtest.append(data[coor[0], coor[1], :])
        Ytest.append(k)

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

# fit data with kernel method

print('Fitting training data now...')
print("--- %s training points ---" % (Xtrain.shape[0]))
start_time = time.time()
D_train = pairwise_distances(Xtrain, Xtrain)     # matrix of distances computed using kernel
sigma = np.sqrt(np.max(D_train))

#clf = svm.SVC(kernel=gaussian_kernel(sigma))

clf = svm.SVC(kernel=laplace_kernel(sigma))
clf.fit(Xtrain, Ytrain)
print('Done with training...')
print("--- %s seconds ---" % (time.time() - start_time))

print('Predicting on test data now...')
start_time = time.time()
Ypredict = clf.predict(Xtest)
print("--- %s seconds ---" % (time.time() - start_time))
print('Perctange of test points labeled incorrectly %s' % (np.round(100*my_discrepancy(Ypredict, Ytest), 1)))