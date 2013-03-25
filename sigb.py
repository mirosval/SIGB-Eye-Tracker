import cv2
from SIGBWindows import SIGBWindows
from SIGBAssignments import *

windows = SIGBWindows(mode="video")

windows.openVideo("Sequences/eye8.avi")
windows.openImage("Sequences/hough2.png")

allTogether(windows)
# glints(windows)
# irisUsingVectors(windows)
# pupilUsingKmeans(windows)
# cannyFitting(windows)
# gradient(windows)
# hough(windows)

# simpleShow(windows)

# windows.show()
windows.show()
