import os
import cv2
import numpy as np
# import time

# reference: https://bretahajek.com/2017/01/scanning-documents-photos-opencv/
def crop_image(image):
    print('Starting To Crop the Image')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and convert to grayscale
    img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserv edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # Median filter clears small details
    img = cv2.medianBlur(img, 11)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    edges = cv2.Canny(img, 200, 250)

    # Getting contours  
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour of biggest rectangle
    # Otherwise return corners of original image
    # Don't forget on our 5px border!
    height = edges.shape[0]
    width = edges.shape[1]
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    # Page fill at least half of image, then saving max area found
    maxAreaFound = MAX_COUNTOUR_AREA * 0.5

    # Saving page contour
    pageContour = np.array([[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])

    # Go through all contours
    for cnt in contours:
        # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound 
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx

    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])
    pageContour = contourOffset(pageContour, (-5, -5))

    # Recalculate to original scale - start Points
    sPoints = pageContour.dot(image.shape[0] / 800)
    
    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                np.linalg.norm(sPoints[3] - sPoints[0]))

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    # Wraping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints) 
    newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

    # resize to fit parsing model (w x h) => (988 x 622)
    h, w, _ = newImage.shape
    new_height = 960
    new_width = 1280
    if h > w:
        # image is rotated longways, so rotate it to landscape
        newImage = cv2.rotate(newImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

    resise_ratio = new_height / w
    newImage = cv2.resize(newImage, (0,0), fx = resise_ratio, fy = resise_ratio)
    # cv2.imwrite(os.path.join(INPUT_DIR, 'oriented_image_color.png'), cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

    # make image binary, to remove background color
    # newImage[(newImage > 140) & (newImage < 250)] = 255
    # newImage[newImage < 250] = 0
    # newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    # (thresh, newImage) = cv2.threshold(newImage, 10, 255,
    #                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # newImage = ~newImage
    # dilated_img = cv2.dilate(newImage, kernel, iterations=1)
    # dilated_img = ~dilated_img

    # Saving the result. Yay! (don't forget to convert colors bact to BGR)
    # cv2.imwrite(os.path.join(INPUT_DIR, 'oriented_image.png'), cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
    print('Finished Orienting the Image')
    return cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)

def resize(img, height=800):
    """ Resize image to given height """	    
    rat = height / img.shape[0]	    
    return cv2.resize(img, (int(rat * img.shape[1]), height))	

def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt