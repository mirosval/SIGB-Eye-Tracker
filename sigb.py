import cv2
from SIGBWindows import SIGBWindows
from SIGBAssignments import *

windows = SIGBWindows(mode="video")

windows.openVideo("Sequences/eye7.avi")
windows.openImage("Sequences/hough2.png")

kmeans(windows)
# cannyFitting(windows)
# gradient(windows)
# hough(windows)

# windows.show()
windows.show(True)
