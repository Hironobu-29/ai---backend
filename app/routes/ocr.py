import easyocr
import cv2
from image_preprocessing import preprocess_image

# Easy-Yolo-OCR の初期化
reader = easyocr.Reader(['en', 'ja'])  # 英語 & 日本語対応

def recognize_text(image_path):
    """ 画像からテキストを抽出する """
    img = preprocess_image(image_path)  # 画像の前処理
    results = reader.readtext(img)  # Easy-Yolo-OCR を適用

    extracted_text = []
    for bbox, text, prob in results:
        if prob > 0.7:  # 70%以上の確信度のみ採用
            extracted_text.append(text)

    return " ".join(extracted_text) if extracted_text else None
