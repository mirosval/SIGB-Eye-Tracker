import cv2
from SIGBWindows import SIGBWindows
from SIGBAssignments import *

windows = SIGBWindows()

windows.openVideo("Sequences/eye1.avi")

kmeans(windows)
# cannyFitting(windows)
# gradient(windows)

windows.show()
