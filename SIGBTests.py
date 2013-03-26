from __future__ import print_function
import cv2
from SIGBSolutions import *

# Testing framework
# Applies the pupil detection to images specified, writes the image with
# pupil, iris, glints drawn on to the Resutls folder in the following format:
#    sequence_frameid.png

# Sequences with representative/challenging frames that we have picked
sequences = {
    "eye1.avi": [8, 87, 223, 424, 582],
    "eye2.avi": [40, 84, 158, 289, 362, 422],
    "eye3.avi": [1, 121, 192, 220, 224, 352, 440, 532],
    "eye4.avi": [26, 123, 156, 192, 234, 273, 354, 402],
    "eye5.avi": [1, 122, 189, 224, 320, 422],
    "eye6.avi": [1, 99, 109, 294, 316, 436, 545],
    "eye7.avi": [1, 51, 102, 146, 184, 218, 224, 237],
    "eye8.avi": [1, 22, 79, 102, 153, 359, 422, 502]  # ,
    # "EyeBizaro": [3, 48, 261, 293, 323, 365, 411, 456] # does not load for some reason?
}

# Total frames processed (useful when a frame can not be loaded,
# we don't want to count it against us
totalFrameCount = 0

# Total frames where the algorithm has detected *something*
# still does not have to be correct detection
totalDetections = 0

# Loop over each sequence
for sequence, frames in sequences.items():
    print("Processing sequence: {0} ({1} frames)".format(sequence, len(frames)), end="")
    video = cv2.VideoCapture("Sequences/" + sequence)

    # Partial frame count for processed frames
    frameCount = 0

    # Partial detection count for processed frames
    detections = 0

    # loop over all frames defined for the sequence at hand
    for frameId in frames:
        print(".", end="")

        # Read the frame
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameId)
        retval, frame = video.read()

        # Erro checking
        if frame == None:
            print("\nFrame Could not be loaded: Retval from video.read(): {} Frame ID: {}".format(retval, frameId))
            continue

        # We dont want to run iris and glint detection on frames
        # we've already drawn in so we make a copy for that
        result = np.copy(frame)

        # detect pupils and draw them
        pupils = getPupils(frame)
        result = drawPupils(result, pupils)

        # cant run iris and glints when no pupil was detected
        if len(pupils) > 0:
            # get and draw iris
            iris = getIrisForPupil(frame, pupils[0])
            result = drawIris(result, iris)

            # get and draw glints
            glints = getGlints(frame, iris)
            result = drawGlints(result, glints)

        # Save Frame
        cv2.imwrite("Results/{}_{}.png".format(sequence, frameId), result)

        # now we know that this frame has been correctly processed
        frameCount += 1
        if len(pupils) > 0:
            detections += 1

    # update cumulative detections and frames
    totalDetections += detections
    totalFrameCount += frameCount
    success = float(detections) / float(frameCount) * 100.0
    print("Got {} detections out of {} frames ({}% success rate)".format(detections, frameCount, success))

# report total results
totalSuccess = float(totalDetections) / float(totalFrameCount) * 100.0
print("--------------------------------------------")
print("Total: Frames: {} Detections: {} ({}% success rate)".format(totalFrameCount, totalDetections, totalSuccess))

