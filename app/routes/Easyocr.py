from easyocr import Reader

# OCR モデルを指定
reader = Reader(['en'])  # 必要に応じて言語コードを設定

# 画像をOCRにかける
text_data = reader.readtext("sample_image.png")

# 結果を出力
for detection in text_data:
    print(f"Detected text: {detection[1]} with confidence: {detection[2]}")
