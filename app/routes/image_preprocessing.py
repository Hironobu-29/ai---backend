import cv2
import numpy as np

def preprocess_image(image_path):
    """OCR 用に画像を前処理する"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # グレースケール化
    img = cv2.GaussianBlur(img, (5, 5), 0)  # ノイズ除去
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # しきい値処理
    img = cv2.resize(img, (640, 640))  # 標準サイズにリサイズ
    return img
