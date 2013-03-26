import cv2
from SIGBWindows import SIGBWindows
from SIGBAssignments import *

# Initialize windows for the Eye tracking lab
# I use monitor with resolution of 1680x1050, so the layout has been optimized for this resolution
# You can change video to either "image" to load a static image, or "cam" to use a webcam
# these haven't been extensively tested, so bugs might occur...
windows = SIGBWindows(mode="video")

# loads video
windows.openVideo("Sequences/eye1.avi")

# loads an image
windows.openImage("Sequences/hough2.png")

# the following functions are defined in SIGBAssignments.py
# their purpose is to register and evaluate sliders and pass the
# parameters to a particular function that can accept it

# this will load all of the detectors, pupil, iris and glint all at the same time
allTogether(windows)

# load glint detector with parameters
# glints(windows)

# load iris detector using gradient images
# irisUsingVectors(windows)

# detect pupil using kmeans
# pupilUsingKmeans(windows)

# canny fitting experiment (not used)
# cannyFitting(windows)

# gradient experiments, used in irisUsingVectors()
# gradient(windows)

# hough experiments, not used, too volatile
# hough(windows)

# just show the image without modification (kind of a template code)
# simpleShow(windows)

# start the show
windows.show()
