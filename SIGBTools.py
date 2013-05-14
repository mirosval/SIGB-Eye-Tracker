import cv2
import numpy as np

from math import *
from pylab import *
from scipy.cluster.vq import *
# from scipy.misc import imresize

# Various tools for use with the eye tracker
# Please note that I copied some of the code from original
# SIGB Tools module from Dan Witzner Hansen, IT University
# I've credited him in the docstrings of the functions that
# I've copied form him

def getGray(image):
    '''
    Wrapper for OpenCV function to convert to grayscale
    
    Params:
        image (numpy array): BGR image as numpy array
        
    Returns:
        (numpy array) grayscale image
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getKMeans(image, featureCount=2, distanceWeight=2, smallSize=(100, 100), show=False):
    '''
    Calculate k-means for the image
    
    Original Author: Dan Witzner Hansen, IT University
    
    Params:
        image (numpy array): Grayscale image to be used
        featureCount (int): how many partitions should be created
        distanceWeight (int): > 0 weight of the position parameters
        smallSize (tuple (int, int)): size of the smaller image to perform kmeans on
        show (bool): Show the kmeans image
    
    Returns:
        (centroids, variance)
    '''
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

def getOrientationAndMagnitude(image, show=False):
    '''
    Calculate orientation and magnitude of the gradient image
    and return it as vector arrays
    
    Uses cv2.fastAtan2 for fast orientation calc, cv2.magnitude
    for fast magnitude calculation
    
    Params:
        image (numpy array): grayscale image to compute this on
        show (bool): show intermediate steps
    
    Returns:
        (orientation, magnitude): numpy arrays
    '''
    sobelHorizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    sobelVertical = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    h = sobelHorizontal
    v = sobelVertical

    orientation = np.empty(image.shape)
    magnitude = np.empty(image.shape)

    height, width = h.shape
    for y in range(height):
        for x in range(width):
            orientation[y][x] = cv2.fastAtan2(h[y][x], v[y][x])

    magnitude = cv2.magnitude(h, v)

    if show:

        fig = figure()
        imshow(magnitude)
        matplotlib.pyplot.show()

        fig2 = figure()
        res = 7
        quiver(h[::res, ::res], -v[::res, ::res])
        imshow(image[::res, ::res], cmap=gray())
        matplotlib.pyplot.show()

    return orientation, magnitude

def getClosed(image, size=5):
    '''
    Morphologically closed image
    
    Performs Morphologic operations erode -> dilate with the same kernel size
    in order to close holes in the image
    
    Args:
        image (Numpy Array): input bitmap image
        size (int): kernel size (1,3,5,7)
    
    Returns:
        filtered bitmap
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image

def getOpen(image, size=5):
    '''
    Morphologically open image
    
    Performs Morphologic operations dilate -> erode with the same kernel size
    in order to close holes in the image
    
    Args:
        image (Numpy Array): input bitmap image
        size (int): kernel size (1,3,5,7)
    
    Returns:
        filtered bitmap
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return image

def applyGradient(image):
    '''
    Apply radial gradient (alpha -> white) from the center of the image
    creating a sort of 'vignette' effect to counter black borders of some
    images that were causing false pupil detects...
    Not used
    
    Params:
        image (numpy array): image to apply the gradient to
    
    Returns:
        image (numpy array) image with the gradient applied
    '''
    if len(image.shape) == 3:
        image = getGray(image)

    height, width = image.shape

    center = (width / 2, height / 2)
    for y, row in enumerate(image):
        for x, value in enumerate(row):
            dx = float(x - center[0]) / float(center[0])
            dy = float(y - center[1]) / float(center[1])
            weight = sqrt((dx ** 2 + dy ** 2) * 0.5)
            image[y][x] = min(255, int(image[y][x] + 255 * weight))

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
    
    Original Author: Dan Witzner Hansen, IT University
    
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

    def getCentroidInt(self):
        centroid = self.getCentroid()

        return (int(centroid[0]), int(centroid[1]))

    def getEquivDiameter(self):
        area = self.getArea()
        return np.sqrt(4 * area / np.pi)

    def getExtend(self):
        area = self.getArea()
        boundingBox = self.getBoundingBox()
        return area / (boundingBox[2] * boundingBox[3])

    def getConvexHull(self):
        return cv2.convexHull(self.contour)

def getCircleSamples(center=(0, 0), radius=1, nPoints=30):
    '''
    Samples a circle with center center = (x,y) , radius =1 and in nPoints on the circle.
    Returns an array of a tuple containing the points (x,y) on the circle and the curve gradient in the point (dx,dy)
    Notice the gradient (dx,dy) has unit length
    
    Original Author: Dan Witzner Hansen, IT University
    
    Params:
        center (tuple (int x, int y)): center of the circle to sample
        radius (int): radius of the circle to sample
        nPoints (int): how many samples do you want
    
    Returns:
        list of samples (tuple containing the points (x,y) on the circle and the curve gradient in the point (dx,dy))
    '''


    s = np.linspace(0, 2 * math.pi, nPoints)
    # points
    P = [(radius * np.cos(t) + center[0], radius * np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s ]
    return P

def getLineCoordinates(p1, p2):
    '''
    Get integer coordinates between p1 and p2 using Bresenhams algorithm
    When an image I is given the method also returns the values of I along the line from p1 to p2. p1 and p2 should be within the image I
    Usage: coordinates=getLineCoordinates((x1,y1),(x2,y2))
    
    Original Auhtor: Dan Witzner Hansen, IT University
    
    Params:
        point1 (tuple(int x,int y)): Start of the line
        point2 (tuple(int x,int y)): End of the line
    
    Returns:
        list of tuples (x,y) coordinates that lie on the line
    '''

    (x1, y1) = p1
    x1 = int(x1); y1 = int(y1)
    (x2, y2) = p2
    x2 = int(x2);y2 = int(y2)

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()

    retPoints = np.array(points)
    X = retPoints[:, 0];
    Y = retPoints[:, 1];


    return retPoints
