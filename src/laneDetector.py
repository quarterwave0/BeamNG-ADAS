import numpy as np

def laneDetection(image, sliceWidth):

    # Histogram computation
    laneBoundaries = []  # [ [leftIndex, rightIndex, midIndex, RowIndex] ]
    #sliceWidth = 10

    for i in range(900 // sliceWidth):
        sliceFrame = image[i * sliceWidth:i * sliceWidth + sliceWidth]
        histo = np.sum(sliceFrame, axis=0)

        try:
            maxL = np.max(np.argwhere(histo[0:400] > 10))  # find the rightmost value that is above ten
        except:
            maxL = 0

        try:
            maxR = np.min(np.argwhere(histo[400:800] > 10)) + 400  # find the leftmost value that is above ten
        except:
            maxR = 0

        maxMid = (maxL + maxR) // 2

        laneBoundaries.append([maxL, maxR, maxMid, i * sliceWidth])

    return laneBoundaries