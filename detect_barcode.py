# python detect_barcode_opencv.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2
import zxing


# construct the argument parse and parse the arguments
image_path = r"C:\Development\tax-form-ocr-docker\app\scans\IMG_1266.jpg"
image = cv2.imread(image_path)
show = False

# load the image and convert it to grayscale

#resize image
image = cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#calculate x & y gradient
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
if show == 1:
	cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# blur the image
blurred = cv2.blur(gradient, (3, 3))

# threshold the image
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

if show == 1:
	cv2.imshow("threshed",cv2.resize(thresh,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

if show == 1:
	cv2.imshow("morphology",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

if show == 1:
	cv2.imshow("erode/dilate",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

sorted_cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
c = sorted_cnts[0]
c1 = sorted_cnts[1]
c2 = sorted_cnts[2]
c3 = sorted_cnts[3]
c4 = sorted_cnts[4]

# compute the rotated bounding box of the largest contours
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
rect1 = cv2.minAreaRect(c1)
box1 = np.int0(cv2.boxPoints(rect1))
rect2 = cv2.minAreaRect(c2)
box2 = np.int0(cv2.boxPoints(rect2))
rect3 = cv2.minAreaRect(c3)
box3 = np.int0(cv2.boxPoints(rect3))
rect4 = cv2.minAreaRect(c4)
box4 = np.int0(cv2.boxPoints(rect4))

# # draw a bounding box arounded the detected barcode and display the
# # # # image
cv2.drawContours(image, [box], -1, (0, 0, 255), 3)
cv2.drawContours(image, [box1], -1, (0, 255, 0), 3)
cv2.drawContours(image, [box2], -1, (255, 0, 0), 3)
cv2.drawContours(image, [box3], -1, (255, 255, 0), 3)
cv2.drawContours(image, [box4], -1, (0, 255, 255), 3)
cv2.imwrite('temp.png', image)
# rois = []
# buffer = 100
# h,w = rect[1]
# # simple heuristic, the image we want should be a long rectangle, so one dimension should be 
# # at least 2x the other
# if h > w *2 or w > h * 2:
#     box = np.clip(box, 0, None)
#     xs = [x[0] for x in box]
#     ys = [x[1] for x in box]
#     min_x = np.min(xs) - buffer
#     min_y = np.min(ys) - buffer
#     max_x = np.max(xs) + buffer
#     max_y = np.max(ys) + buffer
#     rois.append(image[min_y:max_y, min_x:max_x, :])

# h,w = rect1[1]
# if h > w *2 or w > h * 2:
#     box1 = np.clip(box1, 0, None)
#     xs = [x[0] for x in box1]
#     ys = [x[1] for x in box1]
#     min_x = np.min(xs)
#     min_y = np.min(ys)
#     max_x = np.max(xs)
#     max_y = np.max(ys)
#     rois.append(image[min_y:max_y, min_x:max_x, :])


count = 0
for cnts in sorted_cnts[:5]:
    rect = cv2.minAreaRect(cnts)

    h,w = rect[1] # height, width are the second projection in the rect tuple

    # simple heuristic, the image we want should be a long rectangle, so one dimension should be 
    # at least 2x the other
    if h > w *2 or w > h * 2:
        box = np.int0(cv2.boxPoints(rect))
        box = np.clip(box, 0, None)
        xs = [x[0] for x in box]
        ys = [x[1] for x in box]
        min_x = np.min(xs)
        min_y = np.min(ys)
        max_x = np.max(xs)
        max_y = np.max(ys)
        # no idea if image is rotated, so project it into a non-rotated perspective

        # final image should have the width larger than the heigth
        if h > w:
            # swap
            temp = h
            h = w
            w = temp
        # sort by y by descending
        buffer = 100
        sorted_points = sorted(box, key=lambda x: x[1], reverse=True)
        bottom_points = sorted_points[:2]
        top_points = sorted_points[2:]

        # sort by x 
        bottom_points = sorted(bottom_points, key=lambda x: x[0])
        top_points = sorted(top_points, key=lambda x: x[0], reverse=True)

        # add in buffer
        bottom_points[0][0] -= buffer
        bottom_points[0][1] += buffer
        bottom_points[1][0] += buffer
        bottom_points[1][1] += buffer
        top_points[0][0] += buffer
        top_points[0][1] -= buffer
        top_points[1][0] -= buffer
        top_points[1][1] -= buffer

        box = np.array(top_points + bottom_points)
        # box = np.array([[1627, 1201], [1312, 1197], [1299, 2263], [1614, 2267]])
        dst_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        box = box.reshape(-1,1,2)
        M, mask = cv2.findHomography(box, dst_pts, cv2.RANSAC,5.0)
        # matchesMask = mask.ravel().tolist()
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(box,M)
        roi = image[min_y:max_y, min_x:max_x, :]
        size = (int(w), int(h))
        img = cv2.warpPerspective(image, M, size,borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(str(count) + 'temp.png', img)
        reader = zxing.BarCodeReader()
        # barcodes = reader.decode(image_path)
        barcodes = reader.decode(str(count) + 'temp.png', try_harder=True, possible_formats="PDF_417")
        print(barcodes)
        count += 1

# image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# cv2.imshow("Image", image)
# cv2.waitKey(0)