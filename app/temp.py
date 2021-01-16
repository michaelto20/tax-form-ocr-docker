import base64
import cv2

# encode image to base 64
# image = cv2.imread('C:\Development\ocr-document\scans\scan_01.jpg')
# image_encoded = base64.b64encode(image)
# with open ('image_encoded.txt', 'a',) as f:
#     f.write(str(image_encoded))

image = cv2.imread('C:\Development\ocr-document\scans\scan_01.jpg')
image = cv2.resize(image, (0,0), fx = 0.8, fy = 0.8)
cv2.imwrite('C:\Development\ocr-document\scans\scan_01_resized.jpg' , image)