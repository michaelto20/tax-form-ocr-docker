import cv2
import os

TEMPLATES_BASE_DIR = 'templates'
TEMPLATES_DIR = 'w2'
TEMPLATE_NAME = 'form_w2_5.png'

def resize_image(image_path, resize_percentage):
    image = cv2.imread(image_path)
    return cv2.resize(image, (0,0), fx = resize_percentage / 100, fy = resize_percentage / 100)

def save_to_templates_folder(image):
    if TEMPLATE_NAME == '':
        raise Exception("Must set template name")
    if TEMPLATES_BASE_DIR == '':
        raise Exception("Must set template base directory")
    if TEMPLATES_DIR == '':
        raise Exception("Must set template directory")
    template_path = os.path.join(TEMPLATES_BASE_DIR, TEMPLATES_DIR, TEMPLATE_NAME)
    # cv2.imwrite(template_path, image)
    template_path = r"C:/Development/tax-form-ocr-docker/app/templates/w2/form_w2_5_resized.png"
    cv2.imwrite(template_path, image)


if __name__ == "__main__":
    print("Starting to resize image")
    image_path = r"C:/Development/tax-form-ocr-docker/app/templates/w2/form_w2_5.jpg"
    resize_percentage = 125
    resized_image = resize_image(image_path, resize_percentage)
    save_to_templates_folder(resized_image)
    print("Finished resizing image")

