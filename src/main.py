import cv2

from imageFiltering import imgFilter
from laneDetector import laneDetection
from laneCentering import laneCentering
from laneDraw import laneDraw

stream = cv2.VideoCapture(2, cv2.CAP_DSHOW)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Filter parameters
pinch = 400

# Lane detector parameters
sliceWidth = 10

# Lane centering global
v = 0

# Lane centering parameters
targetPoint = 400

while True:
    ret, frame = stream.read()

    filteredFrame, croppedFrame, transformationMatrix = imgFilter(frame, pinch)

    laneBoundaries = laneDetection(filteredFrame, sliceWidth)

    vn = laneCentering(laneBoundaries, targetPoint, v)
    v = vn

    polyFrame, lanes = laneDraw(laneBoundaries, filteredFrame, False, 90000, transformationMatrix, croppedFrame)

    #lanes, points, c, b, vn = pipeline(frame, v)
    #v = vn

    cv2.imshow('Lanes', lanes)
    cv2.setWindowProperty('Lanes', cv2.WND_PROP_TOPMOST, 1)

    cv2.imshow('Birdseye', polyFrame)

    if cv2.waitKey(1) == 27:
        break