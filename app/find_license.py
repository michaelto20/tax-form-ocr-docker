# import the necessary packages
import numpy as np
import cv2
import zxing
import imutils
import concurrent.futures

def use_edge_poly_detection(image):
    image = imutils.resize(image, height=500)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    cv2.imwrite('edges_poly.png', edges)
    # close any lines through dilation so findContours see them better
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 2)
    cv2.imwrite('dilation_poly.png', dilation)
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
        temp = image.copy()
        cv2.drawContours(temp, [dl_Cnt], -1, (0,255,0), 2)
        cv2.imwrite('countours_poly.png', temp)
        return temp
    else:
        return None

    # # if we haven't found a rectangle, just different method
    # c = cnts[0]
    # rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], -1, (0, 0, 255), 3)


def find_drivers_license(image):
    results = []
    license_finder_functions = [use_edge_poly_detection]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_form_templates_path = {executor.submit(func, image): func for func in license_finder_functions}
        for future in concurrent.futures.as_completed(future_form_templates_path):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print('blew up in parallel processing')
                print(exc.message)

    return results


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    image_path = r"C:\Development\tax-form-ocr-docker\app\scans\dl_1.jpg"
    image = cv2.imread(image_path)
    result = find_drivers_license(image)

    # TODO:
    # Use Gaussian blur with
    #   Otsu threshold
    #   Dilation
    #   cv2.minAreaRect(contour) and np.int0(cv2.boxPoints(rect))
    # Distance Transform
    # Method from detect_barcode.py