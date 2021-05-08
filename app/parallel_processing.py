import numpy as np
import cv2
import pytesseract
import math
import pandas as pd
# import random
pd.options.mode.chained_assignment = None  # default='warn'

def process_ocr_location(loc, aligned):
	parsingResults = []
	kernel = np.ones((2,2),np.uint8)

	# extract the OCR ROI from the aligned image
	(x, y, x2, y2) = loc.bbox
	w = x2 - x
	h = y2 - y

	# (x, y, w, h) = loc.bbox
	roi = aligned[y:y + h, x:x + w]

	# try to OCR the image directly
	text = ''
	conf = 0
	if not loc.is_checkbox:
		text, conf = get_text_and_probabilities(roi)

	# successfully OCR image as-is, no need for more preprocessing
	if conf < 75:
		# clean up image lines
		# Apply edge detection method on the image 
		roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi_without_lines = remove_lines(roi_gray)
		# cv2.imwrite('roi_white.png', roi)
		# OCR the ROI using Tesseract
		# print('Binarize Image')
		binarized_roi = binarize_image(roi_without_lines)

		# gray_roi = cv2.cvtColor(binarized_roi, cv2.COLOR_BGR2RGB)
		black_rgb = cv2.bitwise_not(binarized_roi)
		opened_black = cv2.morphologyEx(black_rgb, cv2.MORPH_OPEN, kernel)
		# eroded_black = cv2.erode(opened_black,kernel,iterations = 1)
		# erosion = cv2.erode(rgb,kernel,iterations = 1)
		opened = cv2.bitwise_not(opened_black)
		
		#TODO: Add white border to image to improve OCR
		border_size = 5
		if loc.is_checkbox:
			# pytesseract sucks at parsing just an X for a checkbox, use 20% heuristic instead
			black_pixels = np.count_nonzero(opened==0)
			h,w = opened.shape
			num_pixels = h * w
			black_ratio = (black_pixels * 100) // num_pixels
			if black_ratio >= 20:
				text = 'X'
		else:
			opened_border = cv2.copyMakeBorder(opened,
			border_size,border_size,border_size,border_size,
			borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
			# if opened_border.shape[0] < 75 or opened_border.shape[1] < 75:
			# 	opened_border = cv2.resize(opened_border, (0,0), fx = 2, fy = 2)
			text = pytesseract.image_to_string(opened_border)
		if len(text) == 1 and text != 'X' and ord(text) == 12:
			# ocr'd blank line, so try eroding the image and ocr again
			black_ob = cv2.bitwise_not(opened_border)
			eroded_ob = cv2.erode(black_ob,kernel,iterations = 1)
			white_ob = cv2.bitwise_not(eroded_ob)
			text = pytesseract.image_to_string(white_ob)
		
		# still can't read text, try again
		if len(text) == 1 and text != 'X' and ord(text) == 12 or text.strip() == '':
			closed_roi = cv2.morphologyEx(black_rgb, cv2.MORPH_CLOSE, kernel)
			white_closed_roi = cv2.bitwise_not(closed_roi)

			white_closed_border = cv2.copyMakeBorder(white_closed_roi,
			border_size,border_size,border_size,border_size,
			borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
			# if opened_border.shape[0] < 75 or opened_border.shape[1] < 75:
			# 	opened_border = cv2.resize(opened_border, (0,0), fx = 2, fy = 2)
			text = pytesseract.image_to_string(white_closed_border)

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

def get_text_and_probabilities(roi):
	text = pytesseract.image_to_data(roi, output_type='data.frame')
	text_cleaned = text[text.conf != -1]
	text_cleaned['text'] = text_cleaned['text'].astype(str)

	lines = text_cleaned.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                                     .apply(lambda x: ' '.join(list(x))).tolist()
	# lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
    #                                  .apply(lambda x: ' '.join(x))									 
	confs = text_cleaned.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
		
	# line_conf = []
		
	# for i in range(len(lines)):
	# 	if lines[i].strip():
	# 		line_conf.append((lines[i], round(confs[i],3)))
	average_conf = 0
	return_text = ''
	if len(confs) > 0:
		average_conf = np.array(confs).mean()
		return_text = "".join([c for c in lines]).strip()
	return return_text, average_conf


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