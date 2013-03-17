import cv2

def getGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
