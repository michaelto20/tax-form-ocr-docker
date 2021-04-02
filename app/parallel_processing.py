import numpy as np
import cv2
import pytesseract
import math
import random

def process_ocr_location(loc, aligned):
	parsingResults = []
	kernel = np.ones((2,2),np.uint8)

	# extract the OCR ROI from the aligned image
	(x, y, x2, y2) = loc.bbox
	w = x2 - x
	h = y2 - y

	# (x, y, w, h) = loc.bbox
	roi = aligned[y:y + h, x:x + w]
	# cv2.imwrite(f'output/{random.randint(0,10000000000)}.png', roi)
	# clean up image lines
	# Apply edge detection method on the image 
	roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	roi = remove_lines(roi_gray)
	# cv2.imwrite('roi_white.png', roi)
	# OCR the ROI using Tesseract
	# print('Binarize Image')
	binarized_roi = binarize_image(roi)

	gray_roi = cv2.cvtColor(binarized_roi, cv2.COLOR_BGR2RGB)
	black_rgb = cv2.bitwise_not(gray_roi)
	opened_black = cv2.morphologyEx(black_rgb, cv2.MORPH_OPEN, kernel)
	# erosion = cv2.erode(rgb,kernel,iterations = 1)
	opened = cv2.bitwise_not(opened_black)
	
	#TODO: Add white border to image to improve OCR
	border_size = 5
	opened_border = cv2.copyMakeBorder(opened,
	border_size,border_size,border_size,border_size,
	borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
	text = pytesseract.image_to_string(opened_border)

	# break the text into lines and loop over them
	for line in text.split("\n"):
		# if the line is empty, ignore it
		if len(line) == 0:
			continue

		# convert the line to lowercase and then check to see if the
		# line contains any of the filter keywords (these keywords
		# are part of the *form itself* and should be ignored)
		lower = line.lower()
		count = sum([lower.count(x) for x in loc.filter_keywords])

		# if the count is zero than we know we are *not* examining a
		# text field that is part of the document itself (ex., info,
		# on the field, an example, help text, etc.)
		if count == 0 or loc.filter_keywords[0] == 'force_parse':
			# update our parsing results dictionary with the OCR'd
			# text if the line is *not* empty
			parsingResults.append((loc, line))

	return parsingResults


def binarize_image(gray_image):
	# turn to grayscale first
	# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# remove any major lines still in roi
def remove_lines(roi):
	# detect edges
	edges = get_edges(roi)

	# detect lines
	lines = get_lines(edges)

	if type(lines) == list and len(lines) > 0:
		# white out major lines
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(roi,(x1,y1),(x2,y2),(255,255,255),6)

	# # use make on roi to remove lines
	# masked_roi_and = cv2.bitwise_and(roi, roi, mask=mask)
	# masked_roi_not = cv2.bitwise_not(roi, roi, mask=mask)
	# cv2.imwrite('roi_white.png', roi)
	return roi
	


def create_lines_mask(lines, roi):
	mask = np.zeros_like(roi)

	# use major lines to create mask
	
	for x1,y1,x2,y2 in lines[0]:
		cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),2)

	return mask
	



def get_lines(edges, threshold = 35, min_line_length = 20, max_line_gap = 15):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 360  # angular resolution in radians of the Hough grid

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    return lines

def get_edges(img, min_thresh = 50, max_thresh = 150, kernel = 3):
    edges = cv2.Canny(img, min_thresh, max_thresh, kernel)
    return edges