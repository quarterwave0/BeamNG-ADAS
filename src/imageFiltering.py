import cv2
import numpy as np

whiteLower = np.array([0, 0, 220])
whiteUpper = np.array([80, 10, 255])

shadowLower = np.array([90, 15, 190])
shadowUpper = np.array([179, 255, 255])

#yellowLower = np.array([15, 90, 130])
yellowLower = np.array([15, 90, 190])
yellowUpper = np.array([179, 255, 255])

targetFrame = np.float32([[0, 900], [800, 900], [800, 0], [0, 0]])
def imgFilter(rawImage, pinch):

    # Crop raw image
    croppedImage = rawImage[500:700, 200:1720]

    # Transform
    transformFrame = np.float32([[1220, 200], [300, 200], [300 + pinch, 10], [1220 - pinch, 10]]) # todo: more customization for this

    transformationMatrix = cv2.getPerspectiveTransform(transformFrame, targetFrame)
    transformedFrame = cv2.warpPerspective(croppedImage, transformationMatrix, (800, 900))

    # Prepare frame
    colorTransformFrame = cv2.cvtColor(transformedFrame, cv2.COLOR_BGR2HSV)

    # Blur frame
    blurredFrame = cv2.GaussianBlur(colorTransformFrame, (5, 5), 0)

    # Mask frame
    whiteMask = cv2.inRange(blurredFrame, whiteLower, whiteUpper)
    shadowMask = cv2.inRange(blurredFrame, shadowLower, shadowUpper)
    yellowMask = cv2.inRange(blurredFrame, yellowLower, yellowUpper)

    # Filter frame
    wOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=whiteMask), cv2.COLOR_HSV2BGR)
    yOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=yellowMask), cv2.COLOR_HSV2BGR)
    sOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=shadowMask), cv2.COLOR_HSV2BGR)

    # Finalize frame
    colorOnlyFrame = cv2.cvtColor(cv2.bitwise_or(wOnly, yOnly), cv2.COLOR_BGR2GRAY)
    finalFrame = cv2.bitwise_or(colorOnlyFrame, cv2.cvtColor(sOnly, cv2.COLOR_BGR2GRAY))

    return finalFrame, croppedImage, transformationMatrix
