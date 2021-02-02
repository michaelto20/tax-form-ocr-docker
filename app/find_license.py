# import the necessary packages
import numpy as np
import cv2
import zxing
import imutils
import concurrent.futures
import random as rng
import time
import os
from image_utils import align_images_sift
from dbr import *
from decode_barcode import decode_barcode


def get_roi_from_approxPoly(image, corners, buffer = 10):
    sorted_y_points = sorted(corners, key=lambda x: x[0][1], reverse=True)
    sorted_x_points = sorted(corners, key=lambda x: x[0][0], reverse=True)

    h,w,_ = image.shape
    buffer = 10

    # prevent values from being off image
    max_y = np.clip(sorted_y_points[0][0][1] + buffer, None, h)
    min_y = np.clip(sorted_y_points[3][0][1] - buffer, 0, None)
    max_x = np.clip(sorted_x_points[0][0][0] + buffer, None, w)
    min_x = np.clip(sorted_x_points[3][0][0] - buffer, 0, None)

    return image[min_y:max_y, min_x:max_x,:]

def get_corners_from_contour(cnts):
    # sort by y by descending
    buffer = 100
    sorted_points = sorted(cnts, key=lambda x: x[1], reverse=True)
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

def use_distance_tranform_detection(image):
    img = image.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]  
    cv2.imwrite('watershed.png', img)

def use_edge_poly_detection(image):
    image = imutils.resize(image, height=500)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    # cv2.imwrite('edges_poly.png', edges)
    # close any lines through dilation so findContours see them better
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 2)
    # cv2.imwrite('dilation_poly.png', dilation)
    cnts = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts =  imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    dl_Cnt = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)

        # if it has 4 corners then assume we found the license
        if len(approx) == 4:
            dl_Cnt = approx
            break

    if len(dl_Cnt) > 0:
        return get_roi_from_approxPoly(image, dl_Cnt)
        # temp = image.copy()
        # cv2.drawContours(temp, [dl_Cnt], -1, (0,255,0), 2)
        # cv2.imwrite('countours_poly.png', temp)
        # return None
    else:
        return None

    # # if we haven't found a rectangle, just different method
    # c = cnts[0]
    # rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], -1, (0, 0, 255), 3)

def use_heavy_edge_poly_detection(image):
    image = imutils.resize(image, height=500)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    # cv2.imwrite('heavy_edges_poly.png', edges)
    # close any lines through dilation so findContours see them better
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 2)
    # cv2.imwrite('heavy_dilation_poly.png', dilation)
    cnts = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts =  imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    dl_Cnt = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if it has 4 corners then assume we found the license
        if len(approx) == 4:
            dl_Cnt = approx
            break

    if len(dl_Cnt) > 0:
        return get_roi_from_approxPoly(image, dl_Cnt)

        # temp = image.copy()
        # cv2.drawContours(temp, [dl_Cnt], -1, (0,255,0), 2)
        # cv2.imwrite('heavy_countours_poly.png', temp)
        # return temp
    else:
        return None

def find_drivers_license(image):
    results = []
    license_finder_functions = [use_edge_poly_detection, use_heavy_edge_poly_detection]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_form_templates_path = {executor.submit(func, image): func for func in license_finder_functions}
        for future in concurrent.futures.as_completed(future_form_templates_path):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print('blew up in parallel processing')
                print(exc)

    return results

def read_barcode(image):
    image_path = str(time.time()) + 'temp.png' 
    cv2.imwrite(image_path, image)
    reader = zxing.BarCodeReader()
    barcode = reader.decode(image_path, try_harder=True, pure_barcode=True, possible_formats="PDF_417")
    os.remove(image_path)
    return barcode

def read_barcodes(images):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_form_templates_path = {executor.submit(read_barcode, image): image for image in images}
        for future in concurrent.futures.as_completed(future_form_templates_path):
            print(future.result())

def find_drivers_license_from_template(image, dl_template):
    # dl_template = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\templates\dl\dl_rotated_cleaned.jpg')
    aligned = align_images_sift(image, dl_template, debug=False)
    # cv2.imwrite('aligned.png', aligned)
    return aligned

def find_barcode_from_template(image, barcode_template):
    # barcode_template = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\templates\dl\GA_template.PNG')
    aligned = align_images_sift(image, barcode_template, debug=True)
    # cv2.imwrite('aligned_barcode.png', aligned)
    return aligned    

def find_barcode_from_template_matching(image, barcode_template):
    # barcode_template = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\templates\dl\GA_template.PNG', 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = barcode_template.shape[::-1]
    res = cv2.matchTemplate(image_gray,barcode_template,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(image,top_left, bottom_right, 255, 2)
    # cv2.imwrite('templateMatched.png', image)
    # cv2.imwrite('aligned_barcode.png', aligned)
    return image    

def find_barcode_from_coordinates(image):
    top_left = (521, 111)
    bottom_right = (1946, 532)

    roi = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    # cv2.imwrite('roi_1266.png', roi)
    # cv2.rectangle(image,top_left, bottom_right, 255, 2)
    # cv2.imwrite('coordinateMatched.png', image)
    # cv2.imwrite('aligned_barcode.png', aligned)
    return roi


def sharpen_image(image):
    # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)

    # cv2.imwrite('sharpen.png', sharpen)
    return sharpen

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def make_border(image, borderSize = 10):
    return cv2.copyMakeBorder(
        image,
        top=borderSize,
        bottom=borderSize,
        left=borderSize,
        right=borderSize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

def read_barcode_trial(image):
    license_key = r't0078xQAAALbXgVMIttujaGmBDcm+kZXaxuQNk9asytcZy926MLF3z6gSvwoji/3M5HQ+vJMmZIgUu/zS4HhtQR5r1X2iwEWonvfQ8QADsyla'
    # json_file = r"Please input your own template path"
    # image_path = r"C:\Development\tax-form-ocr-docker\app\scans\jon_dl.jpg"
    # image_path = r'C:\Development\tax-form-ocr-docker\sharpen_barcode.png'
    # image_path = r'C:\Development\tax-form-ocr-docker\sharpen_half.png'

    reader = BarcodeReader()
    reader.init_license(license_key)
    try:
        text_results = reader.decode_buffer(image)
        # text_results = reader.decode_file(image_path)
        if text_results != None:
            for text_result in text_results:
                # print("Barcode Format :")
                # print(text_result.barcode_format_string)
                # print("Barcode Text :")
                # print(text_result.barcode_text)
                # print("Localization Points : ")
                # print(text_result.localization_result.localization_points)
                # print("-------------")
                return text_result.barcode_text
    except BarcodeReaderError as bre:
        print(bre)

def get_drivers_license_info(dl_image, dl_template_image):
    print('find drivers license from template')
    dl = find_drivers_license_from_template(dl_image, dl_template_image)
    dl = sharpen_image(dl)
    barcode = find_barcode_from_coordinates(dl)
    barcode_with_border = make_border(barcode)
    # read_barcodes([barcode_with_border])
    return read_barcode_trial(barcode_with_border)


if __name__ == "__main__":
    image_path = r"C:\Development\tax-form-ocr-docker\app\scans\jon_dl.jpg"
    image = cv2.imread(image_path)

    dl_template_path = r'C:\Development\tax-form-ocr-docker\app\templates\dl\dl_template.png'
    dl_template_image = cv2.imread(dl_template_path)

    barcode_string = get_drivers_license_info(image, dl_template_image)
    decode_barcode(barcode_string)

    # # construct the argument parse and parse the arguments
    # # os.remove('aligned.png')
    # # image_path = r"C:\Development\tax-form-ocr-docker\app\scans\dl_1.jpg"
    # dl_template_path = r'C:\Development\tax-form-ocr-docker\app\templates\dl\dl_template.png'
    # dl_template_image = cv2.imread(dl_template_path)
    # # for i in range(10,-10,-1):
    # #     i = i / 10
    # #     temp_img = dl_template_image.copy()
    # #     rotated = rotate_image(temp_img, i)
    # #     cv2.imwrite(f'dl_template{i}.png', rotated)

    # image_path = r"C:\Development\tax-form-ocr-docker\app\scans\jon_dl.jpg"
    # # image_path = r'C:\Development\tax-form-ocr-docker\app\scans\scan_drivers_license_ga_rotated_180.jpg'

    # image = cv2.imread(image_path)
    # print('find drivers license from template')
    # dl = find_drivers_license_from_template(image, dl_template_image)
    # dl = sharpen_image(dl)
    # barcode = find_barcode_from_coordinates(dl)
    # barcode_with_border = make_border(barcode)
    # # read_barcodes([barcode_with_border])
    # barcode_string = read_barcode_trial(barcode_with_border)
    # decode_barcode(barcode_string)

    # result = find_drivers_license(dl)
    print('finished')
    # TODO:
    # Use Gaussian blur with
    #   Otsu threshold
    #   Dilation
    #   cv2.minAreaRect(contour) and np.int0(cv2.boxPoints(rect))
    # Distance Transform
    # Method from detect_barcode.py