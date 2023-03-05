import cv2
import numpy as np
from matplotlib import pyplot as plt
# menginisialisasi citra asli
image = cv2.imread('image02.png', 0)

# mengatur ukuran kernel filter
kernel_size = 3
# mengatur variasi deviasi standard pada noise Gaussian
deviations = [16, 32, 64, 128]

# melakukan filter rata-rata pada masing-masing citra bernoise
for d in deviations:

    # menambahkan noise Gaussian pada citra asli
    noise = np.random.normal(0, d, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # membuat kernel filter rata-rata
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    # menampilkan citra asli, citra yang diberi noise, dan citra hasil filter
    cv2.imshow('Original Image', image)
    cv2.imshow(f'Noisy Image {d}', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()