import json
import base64
import io
import cv2
from ocr_form import ocr_tax_form
import numpy as np
import time
from pdf2image import convert_from_path
import os

IS_LOCAL = False
TEMP_DIR = '/tmp'
if IS_LOCAL:
	TEMP_DIR = os.path.join('app', TEMP_DIR)

def handler(event, context):
    print('Inside handler')
    form_info = None
    try:
        print('reading image body from json')
        if IS_LOCAL:
            request_body = json.loads(event)
        else:
            request_body = json.loads(event["body"])

        print('decoding image')
        image_base64 = base64.b64decode(request_body['image'])

        image = None
        if "ispdf" in request_body and request_body["ispdf"].lower() == 'true':
            print('image is pdf')
            filename = f'{time.time()}temp.pdf'
            path_to_save = os.path.join(TEMP_DIR, filename)
            print(f'path to save pdf: {path_to_save}')
            f = open(path_to_save, 'wb')
            f.write(image_base64)
            f.close()
            
            print('saved pdf temp file')
            images = convert_from_path(path_to_save)
            image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

            # remove temp pdf file
            os.remove(path_to_save)
        else:
            print('converted image to uint8')
            nparr = np.fromstring(image_base64, np.uint8)

            print('converted image to color')
            image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        print('get form type')
        form_type = request_body['form_type']
        
        print('begin ocr form')
        result, image, form_info = ocr_tax_form(image, form_type, '')

        print('finished ocr form')

    except Exception as e:
        print('threw exception')
        print(f'{e}')
        print(f'{e.args}')
        form_info = "ERROR"


    print('about to send resonse back to api')
    

    body = {
        "message": form_info
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response


if __name__ == "__main__":
    event = {}
    event["ispdf"] = "False"
    event["form_type"] = "w2"

    with open(r"app/scans/W2-TSBooks.png", "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read())
        event["image"] = encoded_string.decode('utf-8')
    
    event = json.dumps(event)
    print(handler(event, None))