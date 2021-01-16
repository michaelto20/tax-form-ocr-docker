import json
import requests

file = r'C:\Development\tax-form-ocr-docker\app\lambda_event.json'
file_handler = open(file).read()

json_data = json.loads(file_handler)

api_endpoint = r'https://buewtq9isg.execute-api.us-east-1.amazonaws.com/dev'

r = requests.post(url = api_endpoint, data = json_data)

print(r)
print(r.text)
# print(json_data)