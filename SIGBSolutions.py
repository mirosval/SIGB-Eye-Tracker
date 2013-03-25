import cv2
from SIGBTools import *
import operator

######################################################################
#
#    Pupil Detection
#
######################################################################

def getPupilCandidates(image):
    def orderPupilCandidates(pupil):
        c = ContourTools(pupil)
        return c.getExtend()

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imageArea = image.shape[0] * image.shape[1]

    candidates = []
    for contour in contours:
        c = ContourTools(contour)
        area = c.getArea()

        # Filter out too small or too big
        if area < imageArea * 0.002 or area > imageArea * 0.3: continue

        candidates.append(c.getConvexHull())

    candidates = sorted(candidates, key=orderPupilCandidates, reverse=True)

    pupils = []
    for candidate in candidates:
        if len(candidate) < 5: continue
        pupil = cv2.fitEllipse(candidate)

        # Filter out too elliptical
        if abs(pupil[1][0] - pupil[1][1]) > 15: continue

        pupils.append(pupil)

    return pupils

def getPupils(image, kmeansFeatureCount=5, kmeansDistanceWeight=14, show=False):
    if kmeansFeatureCount < 3: return []

    gray = getGray(image)
    gray = cv2.equalizeHist(gray)
#    gray = applyGradient(gray)

    centroids, variance = getKMeans(gray, featureCount=kmeansFeatureCount, distanceWeight=kmeansDistanceWeight, smallSize=(100, 75), show=False)

    centroids = sorted(centroids, key=lambda centroid: centroid[0])

    pupils = []
    retval, gray = cv2.threshold(gray, centroids[0][0], 255, cv2.cv.CV_THRESH_BINARY)

    if show:
        cv2.namedWindow("Thresh")
        cv2.imshow("Thresh", gray)

    # Cleanup using closing
    closed = getOpen(gray, 8)
    if show:
        cv2.namedWindow("Closed")
        cv2.imshow("Closed", closed)

    pupils = getPupilCandidates(closed)

    # Iteratively call itself with lower kmeans until something is found or min kmeans is reached
    if len(pupils) == 0:
        return getPupils(image, kmeansFeatureCount=kmeansFeatureCount - 1, kmeansDistanceWeight=kmeansDistanceWeight, show=show)

    return pupils

def drawPupils(image, pupils):
    for pupil in pupils:
        cv2.ellipse(image, pupil, (255, 0, 0))

    if len(pupils) > 0:
        cv2.ellipse(image, pupils[0], (0, 0, 255), 2)
        center = (int(pupils[0][0][0]), int(pupils[0][0][1]))
        cv2.circle(image, center, 1, (0, 255, 0), 2)

    return image

######################################################################
#
#    Iris Detection
#
######################################################################

def getIrisForPupil(image, pupil, show=False):
    gray = getGray(image)
    orientation, magnitude = getOrientationAndMagnitude(gray, show=False)

    center = (int(pupil[0][0]), int(pupil[0][1]))

    pupilRadius = (pupil[1][0] / 2 + pupil[1][1] / 2) / 2
    irisRadius = 5 * pupilRadius

    pupilSamples = getCircleSamples(center, min(irisRadius * 0.5, pupilRadius * 2))
    irisSamples = getCircleSamples(center, irisRadius)

    finalIrisRadiusVotes = dict()

    for sample in range(len(pupilSamples)):
        pupilSample = (int(pupilSamples[sample][0]), int(pupilSamples[sample][1]))
        irisSample = (int(irisSamples[sample][0]), int(irisSamples[sample][1]))

        sob = getLineCoordinates(pupilSample, irisSample)

        sampleVector = (pupilSample[0] - center[0], pupilSample[1] - center[1])

        dist = sqrt(sampleVector[0] ** 2 + sampleVector[1] ** 2)

        angle = degrees(acos(float(sampleVector[1]) / dist))
        angle = cv2.fastAtan2(sampleVector[1], sampleVector[0])

        cnt = 0
        for s in sob:
            try:
                mag = magnitude[s[1] - 1][s[0] - 1]
            except:
                continue

            if mag > 15 and mag < 30:
                ori = orientation[s[1] - 1][s[0] - 1]
                an = angle + ori - 90.0
                if an > 360.0: an -= 360.0

                if an < 3 or an > 357:
                    radius = sqrt((s[0] - center[0]) ** 2 + (s[1] - center[1]) ** 2)
                    # Round radius to tens
                    radius = round(radius / 10.0) * 10.0
                    radius = int(radius)

                    if show:
                        cv2.circle(image, (s[0], s[1]), 2, (255, 255, 0), 2)

                    if radius not in finalIrisRadiusVotes:
                        finalIrisRadiusVotes[radius] = 0

                    finalIrisRadiusVotes[radius] += 1
                    cnt += 1

        if show:
            cv2.line(image, pupilSample, irisSample, (0, 255, 0))

    finalIrisRadius = max(finalIrisRadiusVotes.iteritems(), key=operator.itemgetter(1))[0]

    if show:
        cv2.circle(image, center, finalIrisRadius, (255, 0, 255), 2)
        cv2.imshow("Iris Samples", image)

    return (center, finalIrisRadius)

def drawIris(image, iris):
    center = iris[0]
    radius = iris[1]
    cv2.circle(image, center, radius, (255, 255, 0), 2)

    return image

######################################################################
#
#    Glint Detection
#
######################################################################

def getGlints(image, iris=None, show=False):
    gray = getGray(image)

    centroids, variance = getKMeans(gray, featureCount=5, distanceWeight=14, smallSize=(100, 75), show=show)

    centroids = sorted(centroids, key=lambda centroid: centroid[0], reverse=True)
    retval, gray = cv2.threshold(gray, centroids[0][0] - centroids[0][1], 255, cv2.cv.CV_THRESH_BINARY)

    result = getOpen(gray, 2)
    glints = []

    imageArea = image.shape[0] * image.shape[1]

    contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        c = ContourTools(contour)

        area = c.getArea()
        if area > 0.001 * imageArea: continue

        center = c.getCentroidInt()

        if iris != None:
            irisCenter = iris[0]
            irisRadius = iris[1]
            if sqrt((center[0] - irisCenter[0]) ** 2 + (center[1] - irisCenter[1]) ** 2) > irisRadius:
                continue

        color = (255, 0, 255)
        radius = int(c.getEquivDiameter() / 2)

        cv2.circle(image, center, radius, color, -1)
        glints.append((center, radius))

    if show:
        cv2.imshow("Glints", image)

    return glints

def drawGlints(image, glints):
    for center, radius in glints:
        cv2.circle(image, center, radius, (255, 0, 255), -1)

    return image
