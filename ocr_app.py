from flask import Flask, request, jsonify
from service import ocr_service


app = Flask(__name__)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # OCR処理を呼び出す
    text = ocr_service.extract_text(file)

    return jsonify({'text': text})

if __name__ == "__main__":
    app.run(debug=True)
