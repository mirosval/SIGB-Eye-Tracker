import cv2
from SIGBTools import *
import operator

######################################################################
#
#    Pupil Detection
#
######################################################################

def getPupilCandidates(image):
    '''
    Applies cv2.findContours to binary image, then filters the contours
    and sorts them to get best guesses for pupil locations. Lastly,
    applies ellipse fitting to those contours and returns sorted list
    of ellipses that are good pupil candidates
    
    Params:
        image (numpy array): Binary image
    
    Returns:
        pupils (list): sorted list of pupil ellipses
    '''
    def orderPupilCandidates(pupil):
        c = ContourTools(pupil)
        return c.getExtend()

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imageArea = image.shape[0] * image.shape[1]

    candidates = []
    # first filter contours
    for contour in contours:
        c = ContourTools(contour)
        area = c.getArea()

        # Filter out too small or too big
        if area < imageArea * 0.002 or area > imageArea * 0.3: continue

        candidates.append(c.getConvexHull())

    candidates = sorted(candidates, key=orderPupilCandidates, reverse=True)

    pupils = []
    # second filter ellipses fitted to those contours
    for candidate in candidates:
        if len(candidate) < 5: continue
        pupil = cv2.fitEllipse(candidate)

        # Filter out too elliptical
        if abs(pupil[1][0] - pupil[1][1]) > 15: continue

        pupils.append(pupil)

    return pupils

def getPupils(image, kmeansFeatureCount=5, kmeansDistanceWeight=14, show=False):
    '''
    Given an image perform kmeans detection, threshold it and perform
    blob analysis to determine pupil candidates, and sort them using
    their extend to get most probable pupil location. Lastly apply 
    ellipse fitting to the contour and return an ordered list of good
    pupil candidates
    
    Params:
        image (numpy array): image to perform pupil detection on (BGR)
        kmeansFeatureCount (int): sub param for kmeans
        kmeansDistanceWeight (int): sub param for kmeans
        show (bool): show partial results
    
    Returns:
        list of ellipse positions that are good guess for a pupil location
        
    '''
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
    '''
    Draws ellipses into the image
    
    Params:
        image (numpy array): color image to draw to
        pupils (list ellipses): output of getPupils()
    
    Returns:
        image (numpy array) with pupils drawn in
    '''
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
    '''
    Find the best iris radius for a given pupil. Always assumes there is one,
    so it will very likely return a result. But can also return a None, when
    there is absolutely no indication of an iris.
    
    Params:
        image (numpy array): color image to use
        pupil (ellipse): location of a pupil
        show (bool): show partial resutls
    
    Returns:
        (center, radius) for the detected iris, center is the same as center of the pupil
    '''
    gray = getGray(image)
    orientation, magnitude = getOrientationAndMagnitude(gray, show=False)

    # pupil, and therefore also iris center
    center = (int(pupil[0][0]), int(pupil[0][1]))

    # average radius for pupil (since it is an ellipse)
    pupilRadius = (pupil[1][0] / 2 + pupil[1][1] / 2) / 2
    # max pupil radius will be at most 5 times pupil radius
    irisRadius = 5 * pupilRadius

    # 30 points laying between pupil and iris
    pupilSamples = getCircleSamples(center, min(irisRadius * 0.5, pupilRadius * 2))

    # 30 points laying on a circle that is bigger than iris
    irisSamples = getCircleSamples(center, irisRadius)

    # vote dict for different radii
    finalIrisRadiusVotes = dict()

    # for each sample point in the concentric circle that lies between pupil and iris
    for sample in range(len(pupilSamples)):
        # starting point for a line that goes from in between pupil and iris edge
        pupilSample = (int(pupilSamples[sample][0]), int(pupilSamples[sample][1]))
        # ending point for the line that ends at 5x pupil radius from the pupil center
        irisSample = (int(irisSamples[sample][0]), int(irisSamples[sample][1]))

        # line defined by pupilSample and irisSample points has the direction of
        # the normal for the iris circle

        # points in the image that lay on the line
        lineCoordinates = getLineCoordinates(pupilSample, irisSample)

        # normal vector for the pupil/iris circles
        sampleVector = (pupilSample[0] - center[0], pupilSample[1] - center[1])

        # length of the normal vector
        dist = sqrt(sampleVector[0] ** 2 + sampleVector[1] ** 2)

        # angle of the normal vector
        angle = cv2.fastAtan2(sampleVector[1], sampleVector[0])

        # loop over all the points on the line
        for s in lineCoordinates:
            # sometimes the line is outside the magnitude arrays, in that case just conitnue the loop
            try:
                mag = magnitude[s[1] - 1][s[0] - 1]
            except:
                continue

            # only consider those points that have magnitude greater than 15 but lower than 30
            # since the gradient is a slow one
            if mag > 15 and mag < 30:
                # orientation at the point in question
                ori = orientation[s[1] - 1][s[0] - 1]

                # cleanup the angle so that it is a comparable number to the angle of the line we've
                # obtained earlier
                an = angle + ori - 90.0
                if an > 360.0: an -= 360.0

                # angle difference should be +-3 degrees
                if an < 3 or an > 357:
                    # we have a good sample point with the right magnitude and orientation
                    # calculate the radius of the iris this would correspond to
                    radius = sqrt((s[0] - center[0]) ** 2 + (s[1] - center[1]) ** 2)
                    # Round radius to tens
                    radius = round(radius / 10.0) * 10.0
                    radius = int(radius)

                    # draw the sample that we have used
                    if show:
                        cv2.circle(image, (s[0], s[1]), 2, (255, 255, 0), 2)

                    # add the radius to the vote dict
                    if radius not in finalIrisRadiusVotes:
                        finalIrisRadiusVotes[radius] = 0

                    finalIrisRadiusVotes[radius] += 1

        # draw the line
        if show:
            cv2.line(image, pupilSample, irisSample, (0, 255, 0))

    # very rare, in normal real life images probably won't occur
    if len(finalIrisRadiusVotes) == 0:
        return None

    # order the radius dict by votes and grab the winner
    finalIrisRadius = max(finalIrisRadiusVotes.iteritems(), key=operator.itemgetter(1))[0]

    # draw the winning radius
    if show:
        cv2.circle(image, center, finalIrisRadius, (255, 0, 255), 2)
        cv2.imshow("Iris Samples", image)

    return (center, finalIrisRadius)

def drawIris(image, iris):
    '''
    Draw the iris detected by getIrisForPupil()
    
    Params:
        image (numpy array): image to draw to
        iris (tuple(center tuple(int, int), radius int): iris to draw
    
    Returns:
        image with the iris drawin in
    '''
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
    '''
    Glint detection function finds glints in the iris area of the eye
    
    Params:
        image (numpy array): image to use for detection
        iris (tuple(center tuple(int, int), radius int): iris from getIrisForPupil()
        show (bool): show intermediate results 
    
    Returns:
        list of glint circles
    '''
    gray = getGray(image)

    # compute kmeans
    centroids, variance = getKMeans(gray, featureCount=5, distanceWeight=14, smallSize=(100, 75), show=show)

    # sort, notice reverse=True, we want the brightest parts
    centroids = sorted(centroids, key=lambda centroid: centroid[0], reverse=True)

    # threshold using values obtained by kmeans
    retval, gray = cv2.threshold(gray, centroids[0][0] - centroids[0][1], 255, cv2.cv.CV_THRESH_BINARY)

    # perform morphologic opening
    result = getOpen(gray, 2)
    glints = []

    imageArea = image.shape[0] * image.shape[1]

    contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours
    for contour in contours:
        c = ContourTools(contour)

        area = c.getArea()
        # reject too big
        if area > 0.001 * imageArea: continue

        center = c.getCentroidInt()

        # reject those that lie outside of the iris
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
    '''
    Draw glints obtained by getGlints()
    
    Params:
        image (numpy array): image to draw to
        glints (list): glints
    
    Results:
        image with glints drawn
    '''
    for center, radius in glints:
        cv2.circle(image, center, radius, (255, 0, 255), -1)

    return image
