import cv2

from imageFiltering import imgFilter
from laneDetector import laneDetection
from laneCentering import laneCentering
from laneDraw import laneDraw

stream = cv2.VideoCapture(2, cv2.CAP_DSHOW)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Filter parameters
pinch = 400  # perspective for the transformation

# Lane detector parameters
sliceWidth = 35  # cut the image into slices this tall

# Lane centering parameters
targetPoint = 400  # centering loop target coordinate

while True:
    ret, frame = stream.read()  # get the image

    filteredFrame, croppedFrame, transformationMatrix = imgFilter(frame, pinch)  # filter the image we got

    laneBoundaries = laneDetection(filteredFrame, sliceWidth)  # detect the lanes on the filtered image

    laneCentering(laneBoundaries, targetPoint)  # center the vehicle on the deteced lanes

    polyFrame, lanes = laneDraw(laneBoundaries, filteredFrame, False, 90000, transformationMatrix, croppedFrame)  # draw detected lanes onto image

    cv2.imshow('Lanes', lanes)
    cv2.setWindowProperty('Lanes', cv2.WND_PROP_TOPMOST, 1)

    cv2.imshow('Birdseye', polyFrame)

    if cv2.waitKey(1) == 27:
        break
