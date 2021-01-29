import os
import sys
import json
from typing import List
from dbr import *
import zxing
import numpy as np
import cv2

def preprocess(image):
	# load the image
	# image = cv2.imread(args["image"])

	# #resize image
	# image = cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

	#convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#calculate x & y gradient
	gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
	gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	# blur the image
	blurred = cv2.blur(gradient, (3, 3))

	# threshold the image
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
	thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return thresh

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# # you can change the following variables' value to your own value.
# license_key = r't0078xQAAALbXgVMIttujaGmBDcm+kZXaxuQNk9asytcZy926MLF3z6gSvwoji/3M5HQ+vJMmZIgUu/zS4HhtQR5r1X2iwEWonvfQ8QADsyla'
# json_file = r"Please input your own template path"
image_path = r"C:\Development\tax-form-ocr-docker\0temp.png"
# image_path = r'C:\Development\tax-form-ocr-docker\app\scans\417_1.JPG'
image = cv2.imread(image_path)
image = preprocess(image)
# # image = unsharp_mask(image,amount=2.0, threshold=150)
# kernel_sharpening = np.array([[0,-1,0],
#                               [-1,5,-1,],
#                               [0,-1,0]])

kernel_sharpening = np.array([[-1,-1,-1],
                              [-1,9,-1,],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
# sigma = 2
# kernel_size = (5, 5)
# blurred = cv2.GaussianBlur(image, kernel_size, sigma)
# sharpened = cv2.subtract(image, blurred) * 2
# sharpened = image + sharpened
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
image_bright_contast = apply_brightness_contrast(image,20,20)
cv2.imwrite('temp.png', image)
reader = zxing.BarCodeReader()
# barcodes = reader.decode(image_path)
barcodes = reader.decode(image_path, try_harder=True, possible_formats="PDF_417")
print(barcodes)




# # # you can change the following variables' value to your own value.
# license_key = r't0078xQAAALbXgVMIttujaGmBDcm+kZXaxuQNk9asytcZy926MLF3z6gSvwoji/3M5HQ+vJMmZIgUu/zS4HhtQR5r1X2iwEWonvfQ8QADsyla'
# # json_file = r"Please input your own template path"
# # image_path = r"C:\Development\tax-form-ocr-docker\app\scans\jon_dl.jpg"
# image_path = r'C:\Development\tax-form-ocr-docker\app\scans\dl_1.jpg'

# reader = BarcodeReader()
# reader.init_license(license_key)
# try:
#     text_results = reader.decode_file(image_path)
#     if text_results != None:
#         for text_result in text_results:
#             print("Barcode Format :")
#             print(text_result.barcode_format_string)
#             print("Barcode Text :")
#             print(text_result.barcode_text)
#             print("Localization Points : ")
#             print(text_result.localization_result.localization_points)
#             print("-------------")
# except BarcodeReaderError as bre:
#     print(bre)