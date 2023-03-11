# Tiara Juli Arsita
# F55121053

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load  the image
img = cv2.imread('image02.png', 0)

# Define the filter parameters
d0 = 50      # Cut-off frequency
sigma = 10   # Standard deviation of Gaussian distribution

# Gaussian Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        mask[i,j] = np.exp(-((d**2)/(2*(sigma**2))))
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * mask
img_filtered_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Ideal Highpass Filter
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), d0, 0, -1)
fshift = np.fft.fftshift(np.fft.fft2(img))
fshift_filtered = fshift * mask
img_filtered_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

# Display the results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_filtered_gaussian, cmap='gray')
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_filtered_ideal, cmap='gray')
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
