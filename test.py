import json
import requests
import base64
import cv2
import time

# ********************
# STEP 1: Load in image to ocr
# ********************
# point this to the image you want to ocr
image = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\scans\scan_drivers_license_ga_resized.jpg')
# image = cv2.imread(r'path/to/image/goes/here')

img_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

# ********************************************************
# STEP 2: Replace <form type here> with the form type found below:
# ********************************************************
# form types:
#   drivers license => dl
#   W2 => w2
# example:
#  json_data = {
#     'image': img_string,
#     'form_type': 'w2'
# }
json_data = {
    'image': img_string,
    'form_type': 'dl'
}
json_data = json.dumps(json_data)
# send http request with image and receive response
headers = {"content-type": "application/json"}
api_endpoint = r'https://buewtq9isg.execute-api.us-east-1.amazonaws.com/dev'

start = time.time()
r = requests.post(url = api_endpoint, data = json_data, headers=headers)
end = time.time()
print(r)
print(r.text)
print(f'total time: {end-start}')