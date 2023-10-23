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
        if not set[0] == 0 and not set[1] == 0: # must have both sides of the lane
            pointPairsLeftX.append(set[0])
            pointPairsRightX.append(set[1])
            pointPairsMidX.append(set[2])
            pointPairsY.append(set[3])

            if guidanceDots: # if we want to see only the guidance dots instead of the lines
                pointsFrame = cv2.circle(pointsFrame, (set[0], set[3]), radius=1, color=(0, 0, 255), thickness=1)
                pointsFrame = cv2.circle(pointsFrame, (set[1], set[3]), radius=1, color=(255, 0, 0), thickness=1)
                pointsFrame = cv2.circle(pointsFrame, (set[2], set[3]), radius=1, color=(0, 255, 0), thickness=1)

    # Spline fit
    polyFrame = pointsFrame

    if len(pointPairsMidX) > 5:
        # spline fit
        splineM = sp.UnivariateSpline(pointPairsY, pointPairsMidX, s=smoothingFactor)
        splineL = sp.UnivariateSpline(pointPairsY, pointPairsLeftX, s=smoothingFactor)
        splineR = sp.UnivariateSpline(pointPairsY, pointPairsRightX, s=smoothingFactor)

        dX = np.linspace(250, 850)

        # we linearly map from 250 to 850 on each set of splines, so we get a list of coordinates to draw
        mdY = splineM(dX)
        ldY = splineL(dX)
        rdY = splineR(dX)

        # set them up for cv2 into [[x1, y1], [x2, y2] ...]
        pointsM = np.column_stack((mdY.astype(int), dX.astype(int)))
        pointsL = np.column_stack((ldY.astype(int), dX.astype(int)))
        pointsR = np.column_stack((rdY.astype(int), dX.astype(int)))

        # we need to flip one to make the fill
        pointsRflip = np.flipud(pointsR)
        fillPts = np.concatenate((pointsL, pointsRflip))

        if not guidanceDots:
            polyFrame = cv2.fillPoly(polyFrame, [fillPts], color=(127, 127, 0,))

            polyFrame = cv2.polylines(polyFrame, [pointsM], isClosed=False, color=(127, 0, 127), thickness=5)
            polyFrame = cv2.polylines(polyFrame, [pointsL], isClosed=False, color=(0, 0, 127), thickness=5)
            polyFrame = cv2.polylines(polyFrame, [pointsR], isClosed=False, color=(127, 0, 0), thickness=5)

    # retransform our frame from the birdseye to lane perspective
    expandedFrame = cv2.warpPerspective(polyFrame, np.linalg.inv(transformMatrix), (cropImg.shape[1], cropImg.shape[0]))

    # apply it 50/50 to the frame we already have
    appliedRetLanesFrame = cv2.addWeighted(cropImg, 0.5, expandedFrame, 1.0, 0.0)

    return polyFrame, appliedRetLanesFrame
