import cv2
from SIGBWindows import SIGBWindows
from SIGBAssignments import *

windows = SIGBWindows(mode="video")

windows.openVideo("Sequences/eye1.avi")
windows.openImage("Sequences/hough2.png")

pupilUsingKmeans(windows)
# cannyFitting(windows)
# gradient(windows)
# hough(windows)

# windows.show()
windows.show()
