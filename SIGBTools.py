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

def getClosed(image, size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (2 * size + 1, 2 * size + 1))
    image = cv2.erode(image, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (2 * size + 1, 2 * size + 1))
    image = cv2.dilate(image, kernel)

    return image

class ContourTools:
    '''Class used for getting descriptors of contour-based connected components 
        
    contour: one contour found through cv2.findContours
    
    The following methods can be used:
    
    getArea: Area within the contour  - float 
    getBoundingbox: Bounding box around contour - 4 tuple (topleft.x,topleft.y,width,height) 
    getLength: Length of the contour
    getCentroid: The center of contour: (x,y)
    getMoments: Dictionary of moments: see 
    getPerimiter: Permiter of the contour - equivalent to the length
    getEquivdiameter: sqrt(4*Area/pi)
    getExtend: Ratio of the area and the area of the bounding box. Expresses how spread out the contour is
    getConvexhull: Calculates the convex hull of the contour points
    
    Returns: Dictionary with key equal to the property name
    
    Example: 
         contours, hierarchy = cv2.findContours(I, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
         goodContours = []
         for c in contours:
            contour = ContourTools(c)
            if contour.getArea() > 100 and contour.getArea() < 200
                goodContours.append(contour)
    '''
    def __init__(self, contour):
        self.contour = contour

    def getArea(self):
        return cv2.contourArea(self.contour)

    def getLength(self):
        return cv2.arcLength(self.contour, True)

    def getPerimeter(self):
        return cv2.arcLength(self.contour, True)

    def getBoundingBox(self):
        return cv2.boundingRect(self.contour)

    def getCentroid(self):
        m = cv2.moments(self.contour)

        if(m['m00'] != 0):
            retVal = (m['m10'] / m['m00'], m['m01'] / m['m00'])
        else:
            retVal = (-1, -1)

        return retVal

    def getEquivDiameter(self):
        area = self.getArea()
        return np.sqrt(4 * area / np.pi)

    def getExtend(self):
        area = self.getArea()
        boundingBox = self.getBoundingBox()
        return area / (boundingBox[2] * boundingBox[3])

    def getConvexHull(self):
        return cv2.convexHull(self.contour)
