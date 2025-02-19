import cv2
import numpy as np

def preprocess_image(image_path):
    """OCR �p�ɉ摜��O��������"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # �O���[�X�P�[����
    img = cv2.GaussianBlur(img, (5, 5), 0)  # �m�C�Y����
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # �������l����
    img = cv2.resize(img, (640, 640))  # �W���T�C�Y�Ƀ��T�C�Y
    return img
