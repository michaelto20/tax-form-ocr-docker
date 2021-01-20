from image_utils import align_images, get_image_similarity_score
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import json
import os
import time
import numpy as np
from drivers_license_scanner  import get_drivers_license_info
# from pyzbar.pyzbar import decode, ZBarSymbol
from multiprocessing import Process, Pipe
import concurrent.futures
from parallel_processing import process_ocr_location


config = r'-l eng --oem 1 --psm 6'
SYMBOLS_TO_STRIP = r'!@#$,%^&*()-_+=`~|{}\/?—'
TRANSLATION_TABLE = dict.fromkeys(map(ord, SYMBOLS_TO_STRIP), None)
CONFIG_DIR = 'template_configurations'
TEMPLATES_BASE_DIR = 'templates'
W2_TEMPLATES_DIR = 'w2'
FORM_1099_TEMPLATES_DIR = 'form_1099_MISC'
NO_TEMPLATE_MATCH_DIR = 'no_template_match'
template_similarity_threshold = 220
IS_LOCAL = True
if IS_LOCAL:
	NO_TEMPLATE_MATCH_DIR = os.path.join('app', NO_TEMPLATE_MATCH_DIR)
	TEMPLATES_BASE_DIR = os.path.join('app', TEMPLATES_BASE_DIR)
	CONFIG_DIR = os.path.join('app', CONFIG_DIR)
	pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# create a named tuple which we can use to create locations of the
	# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords", "is_numeric"])



def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	# return "".join([c if ord(c) < 128 else "" for c in text]).strip()
	return text.translate(TRANSLATION_TABLE)

def fix_image_dimensions(image):
	# resize image if too far from optimal size
	# only checking height assumes image dimensions aren't crazy
	optimal_height = 2000	# size in pixels
	print(f'Image shape: {image.shape}')
	h,w,c = image.shape

	if h <= 1900 or h >= 2200:
		print('Changing image\'s size')
		ratio = optimal_height / h
		image = cv2.resize(image, (0,0), fx = ratio, fy = ratio)

	# elif w < 1600 or h < 1600:
	# 	print('Enlarging image')
	# 	image = cv2.resize(image, (0,0), fx = 1.2, fy = 1.2)
	
	return image


def ocr_tax_form(image, form_type, image_file_path):
	# some times images that are too large or too small mess up later processing
	print('Checking image dimensions')
	image = fix_image_dimensions(image)

	# get form ocr configuration
	print("[INFO] getting OCR configuration...")
	template_name = ''
	form_templates_path = ''
	if form_type == "w2":
		form_templates_path = os.path.join(TEMPLATES_BASE_DIR, W2_TEMPLATES_DIR)
	elif form_type == "dl":
		# write to local file for barcode reader
		filename = f'{time.time()}temp.png'
		path_to_save = '../tmp/' + filename
		if IS_LOCAL:
			path_to_save = os.path.join('app', 'tmp', filename)
		print(f"about to write file {filename}")
		cv2.imwrite(path_to_save, image)
		# print(f'file exists after saving to disk: {os.path.exists(filename)}')
		results = get_drivers_license_info(filename,IS_LOCAL)
		os.remove(path_to_save)
		return "success", None, results
	elif form_type == "1099_MISC":
		form_templates_path = os.path.join(TEMPLATES_BASE_DIR, FORM_1099_TEMPLATES_DIR)
	else:
		raise Exception("Form type not implemented")
	print(f'form type: {form_type}')
	# cv2.imwrite('temp_image.png', image)
	# time.sleep(3)
	# get best form template
	template, template_name, best_similarity_score = get_best_template(form_templates_path, image)
	print(f'best template name: {template_name}')
	# check to make sure we found a suitable template, if not save this image so we can make it into a template
	if best_similarity_score <= template_similarity_threshold:
		save_no_template_match(image)
		return "Cannot read the image at this time, please try again later", None

	# get matching config for template
	print('Getting template configs')
	ocr_configs = get_ocr_configs(template_name)

	OCR_LOCATIONS = []
	for key in ocr_configs:
		OCR_LOCATIONS.append(OCRLocation(key, ocr_configs[key]['coord'], ocr_configs[key]['keywords'], ocr_configs[key]['is_numeric']))

	# align the images
	print("[INFO] aligning images...")
	aligned = align_images(image, template, debug=False)

	# initialize a results list to store the document OCR parsing results
	print("[INFO] OCR'ing document...")
	parsingResults = ocr_image_segments(aligned, OCR_LOCATIONS)

	# initialize a dictionary to store our final OCR results
	results = {}

	# loop over the results of parsing the document
	for (loc, line) in parsingResults:
		# grab any existing OCR result for the current ID of the document
		r = results.get(loc.id, None)

		# if the result is None, initialize it using the text and location
		# namedtuple (converting it to a dictionary as namedtuples are not
		# hashable)
		if r is None:
			results[loc.id] = (line, loc._asdict())

		# otherwise, there exists a OCR result for the current area of the
		# document, so we should append our existing line
		else:
			# unpack the existing OCR result and append the line to the
			# existing text
			(existingText, loc) = r
			text = "{}\n{}".format(existingText, line)

			# update our results dictionary
			results[loc["id"]] = (text, loc)

	# loop over the results
	form_info = {}
	for (locID, result) in results.items():
		# unpack the result tuple
		(text, loc) = result
		
		# extract the bounding box coordinates of the OCR location and
		# then strip out non-ASCII text so we can draw the text on the
		# output image using OpenCV
		(x, y, x2, y2) = loc["bbox"]
		w = x2 - x
		h = y2 - y
		clean = cleanup_text(text)

		# draw a bounding box around the text
		cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# loop over all lines in the text
		# display the OCR result to our terminal
		print(loc["id"])
		print("=" * len(loc["id"]))
		for (i, line) in enumerate(text.split("\n")):
			line = cleanup_text(line).strip()
			if len(line) != 0:
				if (loc["is_numeric"] == True and check_is_float(line)) or loc["is_numeric"] == False:
					print("{}".format(line))
					if loc["id"] not in form_info:
						form_info[loc["id"]] = line
					else:
						form_info[loc["id"]] += line
					# draw the line on the output image
					startY = y + (i * 70) + 40
					cv2.putText(aligned, line, (x, startY),
						cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

	# show the input and output images, resizing it such that they fit
	# on our screen
	# cv2.imshow("Input", imutils.resize(image, width=700))
	# cv2.imshow("Output", imutils.resize(aligned, width=700))
	# cv2.imshow("Template", imutils.resize(template, width=700))
	# cv2.waitKey(0)
	return "success", aligned, form_info


def ocr_image_segments(aligned, OCR_LOCATIONS):

	processes = []
	parent_connections = []
	parsingResults = []
	for loc in OCR_LOCATIONS:
		parent_conn, child_conn = Pipe()
		parent_connections.append(parent_conn)

		process = Process(target=process_ocr_location, args=(loc, aligned, child_conn,)) 
		processes.append(process)

	for process in processes:
		process.start()

	# make sure that all processes have finished
	for process in processes:
		process.join()

	# get results back from child process
	for parent_conn in parent_connections:
		parsingResults.append(parent_conn.recv())

	return parsingResults

def save_no_template_match(image):
	file_to_save = os.path.join(NO_TEMPLATE_MATCH_DIR, str(time.time()) + '.png')
	cv2.imwrite(file_to_save, image)

def get_best_template(form_templates_path, image):
	template = None
	best_similarity_score = 0
	template_name = None

	# get all templates of the same form and see which one matches best
	print(form_templates_path)
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
		future_form_templates_path = {executor.submit(parallel_get_best_template, form_templates_path, filename,image): filename for filename in os.listdir(form_templates_path)}
		for future in concurrent.futures.as_completed(future_form_templates_path):
			# url = future_form_templates_path[future]
			try:
				temp = future.result()
				score, candidate_template, filename = temp
				if score > best_similarity_score:
					template = candidate_template
					best_similarity_score = score
					template_name = filename
			except Exception as exc:
				print('blew up in parallel processing')
				print(exc.message)
			
	# for filename in os.listdir(form_templates_path):
	# 	template_path = os.path.join(form_templates_path, filename)
	# 	print(f'template_path: {template_path}')
	# 	candidate_template = cv2.imread(template_path)
	# 	print(f'getting similarity score')
	# 	score = get_image_similarity_score(candidate_template, image)


	print('finished finding best template')
	return template, template_name, best_similarity_score

def parallel_get_best_template(form_templates_path, filename,image):
	template_path = os.path.join(form_templates_path, filename)
	print(f'template_path: {template_path}')
	candidate_template = cv2.imread(template_path)
	print(f'getting similarity score')
	score = get_image_similarity_score(candidate_template, image)
	return score, candidate_template, filename

def check_is_float(text):
	try:
		num = float(text)
		return True
	except:
		return False

def get_ocr_configs(template_name):
	template_name = template_name.split('.')[0]
	config_path = os.path.join(CONFIG_DIR, template_name + "_config.json")
	ocr_configs = None
	with open (config_path, "r") as config_file_handler:
		ocr_configs = config_file_handler.read().replace('\n', '')
	return json.loads(ocr_configs)

if __name__ == "__main__":
	start = time.time()
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image that we'll align to template")
	ap.add_argument("-f", "--form", required=True,
		help="form to parse")
	args = vars(ap.parse_args())
	
	# load the input image and template from disk
	print("[INFO] loading images...")
	image = cv2.imread(args["image"])

	form_type = args["form"]

	result, image, form_info = ocr_tax_form(image, form_type, args["image"])
	print(result)
	print(form_info)
	end = time.time()
	print(f'time: {end - start}')
	try:
		if image.shape[0] != 0:
			cv2.imshow("output", imutils.resize(image, width=500, height=500))
			# cv2.imshow("output", imutils.resize(image, width=2000))
			cv2.waitKey(0)
	except:
		print("No output image")
