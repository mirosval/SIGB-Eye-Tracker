import cv2
import numpy as np
from pylab import *
from math import *
from SIGBTools import *
from SIGBSolutions import *

# This file contains different detectors that we have been partially building
# to get our final solution

# Look for functions such as getPupils() an drawPupils() in SIGBSolutions.py
# Most of the get*() functions have a "show" parameter that can be set to True
# in which case the function will draw intermediate steps as well

def allTogether(windows):
    def callback(image, sliderValues):

        # results image is the one that we draw our solution into, so we don't
        # want to use it for further detections steps
        result = np.copy(image)

        pupils = getPupils(image, show=False)
        result = drawPupils(result, pupils)

        if len(pupils) > 0:
            iris = getIrisForPupil(image, pupils[0], show=False)
            result = drawIris(result, iris)

            glints = getGlints(image, iris)
            result = drawGlints(result, glints)

        return result

    windows.registerOnUpdateCallback("all", callback, "Temp")

def irisUsingVectors(windows):
    def callback(image, sliderValues):
        result = image

        pupils = getPupils(image)
        pupil = pupils[0]

        iris = getIrisForPupil(image, pupil)

        result = drawIris(iris)

        return result

    windows.registerSlider("angle", 0, 360)
    windows.registerOnUpdateCallback("kmeans", callback, "Temp")

def pupilUsingKmeans(windows):
    def callback(image, sliderValues):
        k = int(sliderValues['K'])
        dw = int(sliderValues['dist_weight'])

        result = image

        pupils = getPupils(image, kmeansFeatureCount=k, kmeansDistanceWeight=dw)

        result = drawPupils(result, pupils)

        return result

    windows.registerSlider("K", 4, 100)
    windows.registerSlider("dist_weight", 12, 100)
    windows.registerOnUpdateCallback("kmeans", callback, "Temp")

def glints(windows):
    def glintCallBack(image, sliderValues):
        glints = getGlints(image)

        return image

    windows.registerOnUpdateCallback("glint", glintCallBack, "Temp")

def cannyFitting(windows):
    def callback(image, sliderValues):
        gray = getGray(image)



        thresh1 = int(sliderValues['canny_thresh1'])
        thresh2 = int(sliderValues['canny_thresh2'])
        canny = cv2.Canny(gray, thresh1, thresh2)



        return canny

    windows.registerSlider("canny_thresh1", 100, 255)
    windows.registerSlider("canny_thresh2", 128, 255)
    windows.registerOnUpdateCallback("canny_fitting", callback, "Temp")

def gradient(windows):
    def gradientCallback(image, sliderValues):
        gray = getGray(image)
        sobelHorizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sobelVertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

        h = sobelHorizontal
        v = sobelVertical
#        h = cv2.convertScaleAbs(sobelHorizontal)
#        v = cv2.convertScaleAbs(sobelVertical)

        result = cv2.addWeighted(h, 0.5, v, 0.5, 0)

        orientation = np.empty(gray.shape)
        magnitude = np.empty(gray.shape)

        height, width = h.shape
        for y in range(height):
            for x in range(width):
                orientation[y][x] = atan2(h[y][x], v[y][x]) * 180 / pi
                magnitude[y][x] = sqrt(pow(h[y][x], 2) + pow(v[y][x], 2))

        result = cv2.convertScaleAbs(magnitude)

#        h = orientation * 179 / 255
#        s = np.empty(gray.shape)
#        s.fill(255)
#        v = np.empty(gray.shape)
#        v.fill(255)
#        hsv = np.dstack((h, s, v))
#        hsv = hsv.astype(np.uint8)
#        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#        return result

        fig = figure()
        res = 5
        quiver(h[::-res, ::res], -v[::-res, ::res])
        show()

        return result

    windows.registerOnUpdateCallback("gradient", gradientCallback, "Temp")




def hough(windows):
    def houghCallback(image, sliderValues):
        gray = getGray(image)

        dp = int(sliderValues['hough_dp'])
        minDist = int(sliderValues['hough_min_dist'])
        param1 = int(sliderValues['hough_param1'])
        param2 = int(sliderValues['hough_param2'])
        minRadius = int(sliderValues['hough_min_radius'])
        maxRadius = int(sliderValues['hough_max_radius'])

        blur = cv2.GaussianBlur(gray, (31, 31), 11)
    #    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, None, 10, 350, 50, 155)

#        gray = cv2.Canny(gray, 100, 128)

        circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, dp, minDist, None, param1, param2, minRadius, maxRadius)

        result = image
        if circles is not None:
            circles = circles[0]
            for circle in circles:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                color = (0, 255, 0)
                cv2.circle(result, center, 3, (0, 255, 100))
                cv2.circle(result, center, radius, color)


            circle = circles[0, :]
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            color = (0, 0, 255)
            cv2.circle(result, center, 3, color)
            cv2.circle(result, center, radius, color, 5)

        return result

    windows.registerSlider("hough_dp", 8, 15)
    windows.registerSlider("hough_min_dist", 307, 500)
    windows.registerSlider("hough_param1", 52, 200)
    windows.registerSlider("hough_param2", 447, 1500)
    windows.registerSlider("hough_min_radius", 28, 500)
    windows.registerSlider("hough_max_radius", 110, 500)
    windows.registerOnUpdateCallback("hough", houghCallback, "Temp")

def simpleShow(windows):
    def callback(image, sliderValues):
        gray = getGray(image)

        return gray

    windows.registerOnUpdateCallback("callback", callback, "Temp")
