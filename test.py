import json
import requests
import base64
import cv2

# encode image to base 64
image = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\scans\scan_01_resized.jpg')
# image = cv2.imread(r'C:\Development\tax-form-ocr-docker\app\scans\scan_drivers_license_ga_rotated_180.jpg')
# image_encoded = base64.b64encode(image)
# _, img_encoded = cv2.imencode('.jpg', image)
img_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

json_data = {
    'image': img_string,
    'form_type': 'w2'
}
json_data = json.dumps(json_data)
# encode image as jpeg
# send http request with image and receive response
headers = {"content-type": "application/json"}
# response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

# file = r'C:\Development\tax-form-ocr-docker\app\lambda_event.json'
# f = open(file, 'r', encoding='utf-8')
# file_handler = f.read()

# json_data = json.loads(json_data)

api_endpoint = r'https://buewtq9isg.execute-api.us-east-1.amazonaws.com/dev'
# api_endpoint = r'https://en6rkt1ldox43.x.pipedream.net'

r = requests.post(url = api_endpoint, data = json_data, headers=headers)

print(r)
print(r.text)
# f.close()
# print(json_data)