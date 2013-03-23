
from SIGBTools import *

def getPupils(image, kmeansFeatureCount=4, kmeansDistanceWeight=12):
    gray = getGray(image)
    gray = cv2.equalizeHist(gray)

    centroids, variance = getKMeans(gray, featureCount=kmeansFeatureCount, distanceWeight=kmeansDistanceWeight, smallSize=(100, 75), show=False)

    centroids = sorted(centroids, key=lambda centroid: centroid[0])

    pupils = []
    retval, gray = cv2.threshold(gray, centroids[0][0] - centroids[0][1], 255, cv2.cv.CV_THRESH_BINARY)

    # Cleanup using closing
    gray = getClosed(gray, 5)

    pupils = getPupilCandidates(gray)

    return pupils
