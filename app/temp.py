import base64
import cv2

# # encode image to base 64
# image = cv2.imread(r'C:\Development\tax-form-ocr-docker\app/scans/scan_drivers_license_ga_resized.jpg')
# image_encoded = base64.b64encode(image)
# with open ('image_encoded.txt', 'a',) as f:
#     f.write(str(image_encoded))

image = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\scans\alli_dl.jpg')
image = cv2.resize(image, (0,0), fx = 0.8, fy = 0.8)
cv2.imwrite(r'C:\Development\tax-form-ocr-docker\app\scans\alli_dl_resized.jpg' , image)