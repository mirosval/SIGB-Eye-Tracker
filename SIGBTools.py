import cv2
import numpy as np

from pylab import *
from scipy.cluster.vq import *
from scipy.misc import imresize

def getGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getKMeans(image, featureCount=2, distanceWeight=2, smallSize=(100, 100), show=False):
    height, width = image.shape

    small = cv2.resize(image, smallSize)

    height, width = small.shape
    X, Y = np.meshgrid(range(height), range(width))
    z = small.flatten()
    x = X.flatten()
    y = Y.flatten()

    O = len(x)

    features = np.zeros((O, 3))
    features[:, 0] = z
    features[:, 1] = y / distanceWeight
    features[:, 2] = x / distanceWeight
    features = np.array(features, 'f')

    centroids, variance = kmeans(features, featureCount)

    if show:
        label, distance = vq(features, centroids)
        labelIm = np.array(np.reshape(label, (height, width)))
        f = figure(1)
        imshow(labelIm)
        f.canvas.draw()
        f.show()

    return centroids, variance
