import cv2
import numpy as np
from scipy import interpolate as sp


def laneDraw(laneBoundaries, filtImg, guidanceDots, smoothingFactor, transformMatrix, cropImg):


    # Prepare frame
    pointsFrame = cv2.cvtColor(filtImg, cv2.COLOR_GRAY2BGR)

    # Housekeeping
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

            if guidanceDots:
                pointsFrame = cv2.circle(pointsFrame, (set[0], set[3]), 1, (0, 0, 255), 1)
                pointsFrame = cv2.circle(pointsFrame, (set[1], set[3]), 1, (255, 0, 0), 1)
                pointsFrame = cv2.circle(pointsFrame, (set[2], set[3]), 1, (0, 255, 0), 1)

    # Spline fit
    polyFrame = pointsFrame

    if len(pointPairsMidX) > 5:
        splineM = sp.UnivariateSpline(pointPairsY, pointPairsMidX, s=smoothingFactor)
        splineL = sp.UnivariateSpline(pointPairsY, pointPairsLeftX, s=smoothingFactor)
        splineR = sp.UnivariateSpline(pointPairsY, pointPairsRightX, s=smoothingFactor)

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

    expandedFrame = cv2.warpPerspective(polyFrame, np.linalg.inv(transformMatrix), (cropImg.shape[1], cropImg.shape[0]))
    appliedRetLanesFrame = cv2.addWeighted(cropImg, 0.5, expandedFrame, 1.0, 0.0)

    return polyFrame, appliedRetLanesFrame
