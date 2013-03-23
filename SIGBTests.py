from __future__ import print_function
import cv2
from SIGBSolutions import *

sequences = ["eye1.avi", "eye2.avi", "eye3.avi", "eye4.avi", "eye5.avi"]

for sequence in sequences:
    video = cv2.VideoCapture("Sequences/" + sequence)
    frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    print("Processing sequence: {0} ({1} frames)".format(sequence, frames))

    detections = 0
    for frameId in range(frames):
        retval, frame = video.read()

        results = getPupils(frame)
        if frameId % 10 == 0:
            print(".", end="")

        if len(results) > 0:
            detections += 1

    print("Pupil Detection: Got {} detections out of {} frames".format(detections, frames))
    break

