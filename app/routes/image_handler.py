import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
image = cv2.imread("sample_image.png")

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ノイズ除去（GaussianBlur）
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 閾値処理
_, binary = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 結果を表示（matplotlibを使用）
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(gray, cmap="gray"), plt.title("Grayscale")
plt.subplot(1, 3, 2), plt.imshow(blur, cmap="gray"), plt.title("Blurred")
plt.subplot(1, 3, 3), plt.imshow(binary, cmap="gray"), plt.title("Binarized")
plt.show()
