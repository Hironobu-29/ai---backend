from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from easyocr import Reader

load_dotenv()

app = Flask(__name__)

# MongoDB 設定
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["ai-receptionist"]
collection = db["ocr_results"]

# OCR 設定
reader = Reader(['en', 'ja'], gpu=True)  # 英語と日本語に対応

# 画像の前処理
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # グレースケール化
    image = cv2.resize(image, (800, 800))  # サイズ標準化
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)  # 二値化
    return image

@app.route("/")
def home():
    return "MongoDB Atlas Connected & OCR API Running!"

# OCR 実行 & MongoDB に保存
@app.route("/ocr", methods=["POST"])
def perform_ocr():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)

        # 画像の前処理
        processed_image = preprocess_image(file_path)

        # OCR 解析
        results = reader.readtext(processed_image)

        # 認識結果を文字列に変換
        extracted_text = " ".join([text[1] for text in results])

        # MongoDB に保存
        ocr_data = {"filename": file.filename, "text": extracted_text}
        collection.insert_one(ocr_data)

        return jsonify({"message": "OCR processed!", "text": extracted_text}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# OCR データ取得
@app.route("/ocr_results", methods=["GET"])
def get_ocr_results():
    try:
        data = list(collection.find({}, {"_id": 0}))
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("./uploads", exist_ok=True)  # アップロード用フォルダ作成
    app.run(debug=True)
