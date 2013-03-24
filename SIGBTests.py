from __future__ import print_function
import cv2
from SIGBSolutions import *

sequences = {
    "eye1.avi": [8, 87, 223, 424, 582],
    "eye2.avi": [40, 84, 158, 289, 362, 422],
    "eye3.avi": [1, 121, 192, 220, 224, 352, 440, 532],
    "eye4.avi": [26, 123, 156, 192, 234, 273, 354, 402],
    "eye5.avi": [1, 122, 189, 224, 320, 422],
    "eye6.avi": [1, 99, 109, 294, 316, 436, 545],
    "eye7.avi": [1, 51, 102, 146, 184, 218, 224, 237],
    "eye8.avi": [1, 22, 79, 102, 153, 359, 422, 502]  # ,
    # "EyeBizaro": [3, 48, 261, 293, 323, 365, 411, 456]
}

totalFrameCount = 0
totalDetections = 0
for sequence, frames in sequences.items():
    print("Processing sequence: {0} ({1} frames)".format(sequence, len(frames)), end="")
    video = cv2.VideoCapture("Sequences/" + sequence)

    frameCount = 0
    detections = 0
    for frameId in frames:
        print(".", end="")
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameId)
        retval, frame = video.read()

        if frame == None:
            print("\nFrame Could not be loaded: Retval from video.read(): {} Frame ID: {}".format(retval, frameId))
            continue

        result = np.copy(frame)

        pupils = getPupils(frame)
        result = drawPupils(result, pupils)

        iris = getIrisForPupil(frame, pupils[0])
        result = drawIris(result, iris)

        glints = getGlints(frame, iris)
        result = drawGlints(result, glints)

        # Save Frame
        cv2.imwrite("Results/{}_{}.png".format(sequence, frameId), result)

        frameCount += 1
        if len(results) > 0:
            detections += 1

    totalDetections += detections
    totalFrameCount += frameCount
    success = float(detections) / float(frameCount) * 100.0
    print("\nPupil Detection: Got {} detections out of {} frames ({}% success rate)".format(detections, frameCount, success))
    print()


totalSuccess = float(totalDetections) / float(totalFrameCount) * 100.0
print("--------------------------------------------")
print("Total: Frames: {} Detections: {} ({}% success rate)".format(totalFrameCount, totalDetections, totalSuccess))

