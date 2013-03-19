import cv2
import numpy as np
from pylab import *
from math import *
from SIGBTools import *

def kmeans(windows):
    def callback(image, sliderValues):
        gray = getGray(image)

        k = int(sliderValues['K'])
        dw = int(sliderValues['dist_weight'])

        centroids, variance = getKMeans(gray, featureCount=k, distanceWeight=dw, smallSize=(100, 75), show=False)

        centroids = sorted(centroids, key=lambda centroid: centroid[0])

        # We're hoping that centroids[0] after sorting is the darkest value, so we subtract the variance and
        # that should give us a reliable value (pupil)
        retval, gray = cv2.threshold(gray, centroids[0][0] - centroids[0][1], 255, cv2.cv.CV_THRESH_BINARY)

        # Cleanup using closing
        gray = getClosed(gray, 5)

        return gray

    windows.registerSlider("K", 4, 100)
    windows.registerSlider("dist_weight", 12, 100)
    windows.registerOnUpdateCallback("kmeans", callback, "Temp")

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

        fig = figure()
        res = 5
        quiver(h[::res, ::res], v[::res, ::res])
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

    windows.registerSlider("hough_dp", 2, 10)
    windows.registerSlider("hough_min_dist", 10, 100)
    windows.registerSlider("hough_param1", 10, 100)
    windows.registerSlider("hough_param2", 350, 1500)
    windows.registerSlider("hough_min_radius", 70, 500)
    windows.registerSlider("hough_max_radius", 160, 800)
    windows.registerOnUpdateCallback("hough", houghCallback, "Temp")
