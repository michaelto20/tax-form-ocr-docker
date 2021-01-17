import json
import base64
import io
import cv2
from ocr_form import ocr_tax_form
import numpy as np

def handler(event, context):
    print('Inside handler')

    try:
        print('reading image body from json')
        request_body = json.loads(event["body"])

        print('decoding image')
        image_base64 = base64.b64decode(request_body['image'])

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
        print(f'Error Message: {e.message}')


    print('about to send resonse back to api')
    

    body = {
        "message": form_info
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
