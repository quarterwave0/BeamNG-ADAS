import numpy as np


def laneDetection(image, sliceWidth):
    # Histogram computation
    laneBoundaries = []  # [ [leftIndex, rightIndex, midIndex, RowIndex] ]

    for i in range(900 // sliceWidth):  # cut the image into height based slices
        sliceFrame = image[i * sliceWidth:i * sliceWidth + sliceWidth]  # bottom of slice to the top of the next one
        histo = np.sum(sliceFrame, axis=0)

        try:
            maxL = np.max(np.argwhere(histo[0:400] > 10))  # find the rightmost value that is above ten
        except:
            maxL = 0

        try:
            maxR = np.min(np.argwhere(histo[400:800] > 10)) + 400  # find the leftmost value that is above ten
        except:
            maxR = 0

        # check that they aren't too close, aren't too far, and are on the right areas of their sections
        if maxL < maxR and abs(maxL - maxR) > 90 and abs(maxL - maxR) < 800 and maxL < 1060 and maxR > 460:
            maxMid = (maxL + maxR) // 2  # computing midpoint

            laneBoundaries.append([maxL, maxR, maxMid, i * sliceWidth])  # lane boundaries and the y coordinate

    return laneBoundaries


if __name__ == '__main__':
    import cv2
    from imageFiltering import imgFilter
    from laneDraw import laneDraw

    # stream = cv2.VideoCapture('../dataset/lane1-CA.mp4')
    # stream = cv2.VideoCapture('../dataset/lane2-CA.mp4')
    # stream = cv2.VideoCapture('../dataset/lane3-CA.mp4')rj
    # stream = cv2.VideoCapture('../dataset/lane4-CA.mp4')
    stream = cv2.VideoCapture('../dataset/rl-EA.mp4')

    while True:
        ret, frame = stream.read()

        filteredFrame, croppedFrame, transformationMatrix = imgFilter(frame, 410)

        laneBoundaries = laneDetection(filteredFrame, 10)

        polyFrame, lanes = laneDraw(laneBoundaries, filteredFrame, True, 900000, transformationMatrix, croppedFrame)

        cv2.imshow('Lanes', lanes)
        cv2.imshow('Birdseye', polyFrame)

        if cv2.waitKey(1) == 27:
            break
