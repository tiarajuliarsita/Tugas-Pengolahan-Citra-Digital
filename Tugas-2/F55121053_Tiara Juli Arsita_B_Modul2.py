import cv2

img = cv2. imread("mammogram.tif", 0)

img_1=255 - img


cv2.imshow("original image", img)
cv2.imshow("image negative", img_1)

cv2.waitKey(0)
cv2.destroyAllWindows()