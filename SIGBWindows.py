import cv2
import numpy as np

class SIGBWindows:
    def __init__(self, mode="video"):
        self.updateCallbacks = dict()
        self.sliders = []
        self.mode = mode
        cv2.namedWindow("Settings")
        cv2.namedWindow("Results")
        cv2.namedWindow("Temp")

    def show(self):
        cv2.resizeWindow("Settings", 1000, 450)
        cv2.moveWindow("Settings", 300, 540)

        cv2.resizeWindow("Results", 640, 480)
        cv2.moveWindow("Results", 0, 0)

        cv2.resizeWindow("Temp", 640, 480)
        cv2.moveWindow("Temp", 1030, 0)

        if self.mode == "cam":
            while True:
                key = cv2.waitKey(1)

                self.image = self.getVideoStreamCam()
                self.update()

                if key == 0:
                    break
        else:
            cv2.setTrackbarPos("video_position", "Settings", 1)
            sliderValues = self.getSliderValues()
            self.image = self.getVideoFrame(sliderValues['video_position'])
            self.update()
            key = cv2.waitKey(0)

        cv2.destroyAllWindows()


    def showCam(self):
        cv2.resizeWindow("Settings", 1000, 450)
        cv2.moveWindow("Settings", 300, 540)

        cv2.resizeWindow("Results", 640, 480)
        cv2.moveWindow("Results", 0, 0)

        cv2.resizeWindow("Temp", 640, 480)
        cv2.moveWindow("Temp", 1030, 0)


        self.update()

        key = cv2.waitKey(0)

    def update(self, trackbarPos=None):
        sliderValues = self.getSliderValues()

        if self.mode == "video":
            sliderValues = self.getSliderValues()
            self.image = self.getVideoFrame(sliderValues['video_position'])

        image = np.copy(self.image)

        cv2.imshow("Results", image)
        cv2.imshow("Temp", image)

        for callbackName in self.updateCallbacks:
            callback = self.updateCallbacks[callbackName]
            func = callback['function']
            window = callback['window']
            result = func(image, sliderValues)

            x = 10
            y = 20
            for slider in sliderValues:
                value = slider + ": " + str(sliderValues[slider])
                cv2.putText(result, value, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
                y = y + 20

            cv2.imshow(window, result)

    def registerSlider(self, name, startingValue, maxValue):
        self.sliders.append(name)
        cv2.createTrackbar(name, "Settings", startingValue, maxValue, self.update)

    def deregisterSlider(self, name):
        self.sliders = [slider for slider in self.sliders if slider != name]

    def getSliderValues(self):
        values = dict()
        for slider in self.sliders:
            values[slider] = cv2.getTrackbarPos(slider, "Settings")
        return values

    def registerOnUpdateCallback(self, name, function, window="Results"):
        self.updateCallbacks[name] = {
                                      'function': function,
                                      'window': window
                                      }

    def openVideo(self, videoFile):
        self.video = cv2.VideoCapture(videoFile)
        self.registerSlider("video_position", 2, self.getTotalVideoFrames())

    def openImage(self, imageFile):
        self.image = cv2.imread(imageFile)


    def getVideoFrame(self, frameIndex):
        frameIndex = min(frameIndex, self.getTotalVideoFrames() - 1)
        self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameIndex)
        retval, image = self.video.read()
        return image

    def getVideoStreamCam(self):
        self.video = cv2.VideoCapture(1)
        retval, image = self.video.read()
        return image

    def getTotalVideoFrames(self):
        return int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

