import numpy as numpy
import cv2
from barcode_reader.reader import BarCodeReader
import os


def get_drivers_license_info(image_path, is_local):
# def get_drivers_license_info(image_path):
    print('inside get_drivers_license_info')
    print(f'does image still exist: {os.path.exists("../tmp/" + image_path)}')
    reader = BarCodeReader(is_local)
    # dlstring = reader.decode_image(image_path)
    dlstring = reader.decode(image_path)

    dlstringarray = dlstring[0]['parsed'].split('\n')
    dlstringarray = dlstringarray[2:]
    dlstringarray = [line.strip() for line in dlstringarray]

    # remove 'ANSI' from first element (It's a fixed header)
    dlstringarray[0] = dlstringarray[0][5:]

    metadata = dlstringarray[0]

    dlstringarray.remove(metadata)

    dl_info = {}

    for field in dlstringarray:

        fieldID = field[0:3]
        fieldValue = field[3:]

        if fieldID == 'DBA':
            dl_info['expiration_date'] = fieldValue
        elif fieldID == 'DCS':
            dl_info['last_name'] = fieldValue
        elif fieldID == 'DCT':
            dl_info['first_name'] = fieldValue
        elif fieldID == 'DAC':
            dl_info['first_name'] = fieldValue
        elif fieldID == 'DBD':
            dl_info['issue_date'] = fieldValue
        elif fieldID == 'DBB':
            dl_info['dob'] = fieldValue
        elif fieldID == 'DBC':
            if fieldValue == '1':
                dl_info['gender'] = 'M' # 1 for male, 2 for female
            else:
                dl_info['gender'] = 'F' # 1 for male, 2 for female
        elif fieldID == 'DAG':
            dl_info['address_line1'] = fieldValue
        elif fieldID == 'DAI':
            dl_info['city'] = fieldValue
        elif fieldID == 'DAJ':
            dl_info['state'] = fieldValue
        elif fieldID == 'DAK':
            # handle case when zip code is in the format XXXXX-XXXX
            zip_code_tokens = fieldValue.split('-')
            if len(zip_code_tokens) == 1:
                dl_info['zip_code'] = fieldValue
            else:
                dl_info['zip_code'] = zip_code_tokens[0]
        elif fieldID == 'DCG':
            dl_info['country'] = fieldValue
        elif fieldID == 'DAA':
            # Full name
            names = fieldValue.split(',')
            if len(names) > 1:
                dl_info['last_name'] = names[0]
                dl_info['first_name'] = names[1]
            elif len(names) == 1:
                dl_info['last_name'] = names[0]
        elif fieldID == 'DAQ':
            dl_info['dl_number'] = fieldValue

        
    print("[INFO] finished reading Driver's License 2D barcode")
    return dl_info