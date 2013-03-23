import cv2
from SIGBTools import *

def getPupils(image, kmeansFeatureCount=5, kmeansDistanceWeight=14, show=False):
    gray = getGray(image)
    gray = cv2.equalizeHist(gray)

    centroids, variance = getKMeans(gray, featureCount=kmeansFeatureCount, distanceWeight=kmeansDistanceWeight, smallSize=(100, 75), show=show)

    centroids = sorted(centroids, key=lambda centroid: centroid[0])

    pupils = []
    retval, gray = cv2.threshold(gray, centroids[0][0] - centroids[0][1], 255, cv2.cv.CV_THRESH_BINARY)

    # Cleanup using closing
    closed = getClosed(gray, 5)
    if show:
        cv2.namedWindow("Closed")
        cv2.imshow("Closed", closed)

    pupils = getPupilCandidates(closed)

    if len(pupils) == 0:
        if show:
            cv2.namedWindow("Threshold")
            cv2.imshow("Threshold", gray)
        pupils = getPupilCandidates(gray)

    return pupils

def getIrisForPupil(image, pupil, show=False):
    orientation, magnitude = getOrientationAndMagnitude(image)

    center = (int(pupil[0][0]), int(pupil[0][1]))

    pupilRadius = (pupil[1][0] / 2 + pupil[1][1] / 2) / 2
    irisRadius = 5 * pupilRadius

    pupilSamples = getCircleSamples(center, min(irisRadius * 0.7, pupilRadius * 2))
    irisSamples = getCircleSamples(center, irisRadius)

    finalIrisRadiusVotes = dict()

    for sample in range(len(pupilSamples)):
        pupilSample = (int(pupilSamples[sample][0]), int(pupilSamples[sample][1]))
        irisSample = (int(irisSamples[sample][0]), int(irisSamples[sample][1]))

        sob = getLineCoordinates(pupilSample, irisSample)

        sampleVector = (pupilSample[0] - center[0], pupilSample[1] - center[1])

        dist = sqrt(sampleVector[0] ** 2 + sampleVector[1] ** 2)

        angle = degrees(acos(float(sampleVector[1]) / dist))

        if sampleVector[0] < 0:
            angle = -1 * angle

        cnt = 0
        for s in sob:
            mag = magnitude[s[1]][s[0]]
            if mag > 15:
                ori = orientation[s[1]][s[0]]
                if abs(angle - ori) < 5:
                    if show:
                        cv2.circle(image, (s[0], s[1]), 2, (255, 255, 0), 2)
                    radius = int(sqrt((s[0] - center[0]) ** 2 + (s[1] - center[1]) ** 2))
                    if radius not in finalIrisRadiusVotes:
                        finalIrisRadiusVotes[radius] = 0

                    finalIrisRadiusVotes[radius] += 1
                    cnt += 1

        if show:
            cv2.line(image, pupilSample, irisSample, (0, 255, 0))

    finalIrisRadiusVotes = sorted(finalIrisRadiusVotes)
    finalIrisRadius = finalIrisRadiusVotes.pop()

    if show:
        cv2.circle(image, center, radius, (255, 0, 255), 2)

    return (center, radius)

