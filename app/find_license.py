# import the necessary packages
import numpy as np
import cv2
import zxing
import imutils
import concurrent.futures
import random as rng
import time
import os

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
    # barcodes = reader.decode(image_path)
    barcode = reader.decode(image_path, try_harder=True, possible_formats="PDF_417")
    os.remove(image_path)
    return barcode

def read_barcodes(images):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_form_templates_path = {executor.submit(read_barcode, image): image for image in images}
        for future in concurrent.futures.as_completed(future_form_templates_path):
            print(future.result())




if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    image_path = r"C:\Development\tax-form-ocr-docker\app\scans\dl.jpg"
    # image_path = r"C:\Development\tax-form-ocr-docker\app\scans\IMG_1269.jpg"
    image = cv2.imread(image_path)
    result = find_drivers_license(image)
    read_barcodes(result)
    print('finished')
    # TODO:
    # Use Gaussian blur with
    #   Otsu threshold
    #   Dilation
    #   cv2.minAreaRect(contour) and np.int0(cv2.boxPoints(rect))
    # Distance Transform
    # Method from detect_barcode.py