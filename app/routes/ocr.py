import easyocr
import cv2
from image_preprocessing import preprocess_image

# Easy-Yolo-OCR �̏�����
reader = easyocr.Reader(['en', 'ja'])  # �p�� & ���{��Ή�

def recognize_text(image_path):
    """ �摜����e�L�X�g�𒊏o���� """
    img = preprocess_image(image_path)  # �摜�̑O����
    results = reader.readtext(img)  # Easy-Yolo-OCR ��K�p

    extracted_text = []
    for bbox, text, prob in results:
        if prob > 0.7:  # 70%�ȏ�̊m�M�x�̂ݍ̗p
            extracted_text.append(text)

    return " ".join(extracted_text) if extracted_text else None
