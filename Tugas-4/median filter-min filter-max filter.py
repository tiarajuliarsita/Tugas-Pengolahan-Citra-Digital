import cv2

# membaca citra asli
img = cv2.imread('image01.png')

# pengolahan citra dengan median filter
median = cv2.medianBlur(img, 5)

# pengolahan citra dengan max filter
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
max = cv2.dilate(img, kernel)

# pengolahan citra dengan min filter
min = cv2.erode(img, kernel)

# menampilkan hasil pengolahan citra
cv2.imshow('Citra Asli', img)
cv2.imshow('Median Filter', median)
cv2.imshow('Max Filter', max)
cv2.imshow('Min Filter', min)

cv2.waitKey(0)
cv2.destroyAllWindows()
