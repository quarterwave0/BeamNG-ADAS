import cv2
import numpy as np
from scipy import interpolate as sp
from scipy import signal as f
import vgamepad as vg
from simple_pid import PID

Kp = 0.0057
#Ki = 0.0005
Ki = 0.0006
#Kd = 0.00195
Kd = 0.00205

p = PID(Kp, Ki, Kd, setpoint=0)
p.output_limits = (-0.3, 0.3)
v = 0

deviationBuffer = []
lowpass = f.butter(3, 0.85, output='sos')

stream = cv2.VideoCapture(2, cv2.CAP_DSHOW)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
gamepad = vg.VX360Gamepad()

gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
gamepad.update()

def pipeline(rawImage, v):
    #Remove Extraneous Information
    croppedImage = rawImage[500:700, 200:1720]

    #Transform
    pinch = 400
    transformFrame = np.float32([[1220, 200], [300, 200], [300+pinch, 10], [1220-pinch, 10]])
    targetFrame = np.float32([[0, 900], [800, 900], [800, 0], [0, 0]])

    transformationMatrix = cv2.getPerspectiveTransform(transformFrame, targetFrame)
    transformedFrame = cv2.warpPerspective(croppedImage, transformationMatrix, (800, 900))

    #Color filters
    colorTransformFrane = cv2.cvtColor(transformedFrame, cv2.COLOR_BGR2HSV)

    blurredFrame = cv2.GaussianBlur(colorTransformFrane, (5, 5), 0)

    whiteLower = np.array([0, 0, 220])
    whiteUpper = np.array([80, 10, 255])

    shadowLower = np.array([90, 15, 190])
    shadowUpper = np.array([179, 255, 255])

    yellowLower = np.array([15, 90, 130])
    yellowUpper = np.array([179, 255, 255])

    whiteMask = cv2.inRange(blurredFrame, whiteLower, whiteUpper)
    shadowMask = cv2.inRange(blurredFrame, shadowLower, shadowUpper)
    yellowMask = cv2.inRange(blurredFrame, yellowLower, yellowUpper)

    wOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=whiteMask), cv2.COLOR_HSV2BGR)
    yOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=yellowMask), cv2.COLOR_HSV2BGR)
    sOnly = cv2.cvtColor(cv2.bitwise_and(blurredFrame, blurredFrame, mask=shadowMask), cv2.COLOR_HSV2BGR)

    colorOnlyFrame = cv2.cvtColor(cv2.bitwise_or(wOnly, yOnly), cv2.COLOR_BGR2GRAY)
    colorFilteredFrame = cv2.bitwise_or(colorOnlyFrame, cv2.cvtColor(sOnly, cv2.COLOR_BGR2GRAY))

    #Histogram computation
    laneBoundaries = [] #[ [leftIndex, rightIndex, midIndex, RowIndex] ]
    sliceWidth = 10

    for i in range(900 // sliceWidth):
        sliceFrame = colorFilteredFrame[i * sliceWidth:i * sliceWidth + sliceWidth]
        histo = np.sum(sliceFrame, axis=0)

        try:
            maxL = np.max(np.argwhere(histo[0:400] > 10))  #find the rightmost value that is above ten
        except:
            maxL = 0

        try:
            maxR = np.min(np.argwhere(histo[400:800] > 10)) + 400  # find the leftmost value that is above ten
        except:
            maxR = 0

        maxMid = (maxL + maxR) // 2

        laneBoundaries.append([maxL, maxR, maxMid, i*sliceWidth])

    pointsFrame = cv2.cvtColor(colorFilteredFrame, cv2.COLOR_GRAY2BGR)
    pointPairsLeftX = []
    pointPairsRightX = []
    pointPairsMidX = []
    pointPairsY = []


    for set in laneBoundaries:
        if not set[0] == 0 and not set[1] == 0:
            pointPairsLeftX.append(set[0])
            pointPairsRightX.append(set[1])
            pointPairsMidX.append(set[2])
            pointPairsY.append(set[3])

            #pointsFrame = cv2.circle(pointsFrame, (set[0], set[3]), 1, (0, 0, 255), 1)
            #pointsFrame = cv2.circle(pointsFrame, (set[1], set[3]), 1, (255, 0, 0), 1)
            #pointsFrame = cv2.circle(pointsFrame, (set[2], set[3]), 1, (0, 255, 0), 1)

    #spline fit
    polyFrame = pointsFrame
    smoothingFactor = 90000
    edgeSmoothingFactor= 90000

    if len(pointPairsMidX) > 5:

        splineM = sp.UnivariateSpline(pointPairsY, pointPairsMidX, s=smoothingFactor)
        splineL = sp.UnivariateSpline(pointPairsY, pointPairsLeftX, s=edgeSmoothingFactor)
        splineR = sp.UnivariateSpline(pointPairsY, pointPairsRightX, s=edgeSmoothingFactor)

        dX = np.linspace(250, 850)

        mdY = splineM(dX)
        ldY = splineL(dX)
        rdY = splineR(dX)

        pointsM = np.array(list(zip(mdY.astype(int), dX.astype(int))))
        pointsL = np.array(list(zip(ldY.astype(int), dX.astype(int))))
        pointsR = np.array(list(zip(rdY.astype(int), dX.astype(int))))

        pointsLFlip = np.flipud(pointsR)
        fillPts = np.concatenate((pointsL, pointsLFlip))

        polyFrame = cv2.fillPoly(polyFrame, [fillPts], color=(127, 127, 0,))

        polyFrame = cv2.polylines(polyFrame, [pointsM], isClosed=False, color=(127, 0, 127), thickness=5)
        polyFrame = cv2.polylines(polyFrame, [pointsL], isClosed=False, color=(0, 0, 127), thickness=5)
        polyFrame = cv2.polylines(polyFrame, [pointsR], isClosed=False, color=(127, 0, 0), thickness=5)

    #drive car
    deviation = np.mean(pointPairsMidX) - 400 #target of 400

    if not np.isnan(deviation):

        deviationBuffer.insert(0, deviation)
        if (len(deviationBuffer) > 20):
            deviationBuffer.pop()

        filteredBuffer = f.sosfilt(lowpass, deviationBuffer)

        corrector = p(v)
        pO, iO, dO = p.components
        print(corrector, filteredBuffer[0], pO, iO, dO)

        gamepad.left_joystick_float(x_value_float=corrector, y_value_float=0.0)

        gamepad.update()
        vn = filteredBuffer[0]

        # if(deviation<-15):
        #     print(deviation, "deviating left")
        #     gamepad.left_joystick_float(x_value_float=0.3, y_value_float=0.0)
        #     gamepad.update()
        # elif (deviation>15):
        #     print(deviation, "deviating right")
        #     gamepad.left_joystick_float(x_value_float=-0.3, y_value_float=0.0)
        #     gamepad.update()
    else:
        vn = v
        gamepad.reset()
        gamepad.update()


    #retransform
    expandedFrame = cv2.warpPerspective(polyFrame, np.linalg.inv(transformationMatrix), (croppedImage.shape[1], croppedImage.shape[0]))
    lanes = cv2.addWeighted(croppedImage, 0.5, expandedFrame, 1.0, 0.0)

    return lanes, pointsFrame, croppedImage, transformedFrame, vn

while True:
    ret, frame = stream.read()

    lanes, points, c, b, vn = pipeline(frame, v)
    v = vn

    cv2.imshow('Lanes', lanes)
    cv2.setWindowProperty('Lanes', cv2.WND_PROP_TOPMOST, 1)

    cv2.imshow('Points', points)
    #cv2.imshow('Cropped', c)
    #cv2.imshow('Birdseye', b)

    if cv2.waitKey(1) == 27:
        break