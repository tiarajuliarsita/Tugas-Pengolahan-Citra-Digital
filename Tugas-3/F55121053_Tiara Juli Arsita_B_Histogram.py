import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca citra
image = cv2.imread('image03.png', 0)

# membuat histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# menampilkan histogram sebelum perbaikan
plt.hist(image.flatten(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()
# mencari nilai intensitas piksel dengan frekuensi terbanyak
max_intensity = np.argmax(hist)

# membuat lookup table untuk menggeser nilai intensitas piksel
lut = np.zeros((256,), dtype=np.uint8)
for i in range(256):
    lut[i] = np.uint8(np.clip((i - max_intensity) * 1.5 + max_intensity, 0, 255))

# mengaplikasikan lookup table ke citra
rst = cv2.LUT(image, lut)

# membuat histogram setelah perbaikan
hist_result, bins_result = np.histogram(rst.flatten(), 256, [0, 256])

# menampilkan histogram setelah perbaikan
plt.hist(rst.flatten(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()
# menampilkan citra asli dan hasil perbaikan
cv2.imshow('Original image', image)
cv2.imshow('Histogram image', rst)

cv2.waitKey(0)
cv2.destroyAllWindows()