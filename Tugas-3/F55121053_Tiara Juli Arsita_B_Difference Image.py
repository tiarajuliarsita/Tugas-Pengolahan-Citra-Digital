import cv2

# membaca citra asli
img = cv2.imread('image01.png', 0)

# membuat citra difference
diff_img = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), img)

# histogram equalization untuk citra difference
equalized_diff_img = cv2.equalizeHist(diff_img)

# menampilkan citra asli, difference image, dan Equalization Difference image
cv2.imshow('Original Image', img)
cv2.imshow('Difference Image', diff_img)
cv2.imshow('Equalized Difference Image', equalized_diff_img)


cv2.waitKey(0)
cv2.destroyAllWindows()