import cv2
from skimage.feature import hog

image = cv2.imread('dataset/A/0.jpg')

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.imshow("test",hog_image)
cv2.waitKey(0)
cv2.destroyAllWindows()