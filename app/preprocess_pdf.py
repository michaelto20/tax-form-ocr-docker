from pdf2image import convert_from_path
import os
import cv2
import numpy as np


BASE_DER = 'app' 
TEMPLATE_DIR = os.path.join(BASE_DER, 'scans')
TEMPLATE_TYPE = os.path.join(TEMPLATE_DIR,'form_1040')

def convert_pdf_template_to_image(pdf_filename, start_template_count):
    print('saved pdf temp file')
    images = convert_from_path(pdf_filename)
    for i in images:
        image = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
        template_name = f'form_1040_{start_template_count}.png'
        cv2.imwrite(os.path.join(TEMPLATE_TYPE, template_name), image)
        start_template_count += 1

if __name__ == "__main__":
    start_template_count = 6
    pdf_filename_to_convert = r'C:\Users\Michael Townsend\Downloads\rapidOCR_images\1040\TaxAct_1040.pdf'
    convert_pdf_template_to_image(pdf_filename_to_convert, start_template_count)