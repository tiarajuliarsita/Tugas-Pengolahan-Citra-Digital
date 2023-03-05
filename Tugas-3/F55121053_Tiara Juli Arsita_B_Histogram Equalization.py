import cv2
import numpy as np
from matplotlib import pyplot as plt
# membaca citra
image = cv2.imread('image04.png', 0)

# membuat  histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# menampilkan histogram sebelum perbaikan
plt.hist(image.flatten(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()

# melakukan ekualisasi histogram
equalized_img = cv2.equalizeHist(image)

# membuat histogram setelah perbaikan
hist_result, bins_result = np.histogram(equalized_img.flatten(), 256, [0, 256])

# menampilkan histogram setelah perbaikan
plt.hist(equalized_img.flatten(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()

# menampilkan citra asli dan hasil perbaikan
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()