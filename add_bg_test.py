import numpy as np
import cv2
import segmentation

image = cv2.imread('dataset/B/0.jpg')
bg = cv2.imread('bg_change/bg12.jpg')
bg = cv2.resize(bg,(128,128))


# bg = np.random.randint(0,255,size=(128,128,3))
# bg = np.array(bg,dtype=np.uint8)
# print(bg)
# cv2.imshow("test",bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# alpha = 0.8

# dst = cv2.addWeighted(image, alpha , bg, 1-alpha, 0)

# cv2.namedWindow("test",cv2.WINDOW_NORMAL)
# cv2.imshow("test",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

out1 = segmentation.segment(image)
out1 = cv2.bitwise_and(image,image,mask=out1)
print(out1)
cv2.imshow("test",out1)
cv2.waitKey(0)
cv2.destroyAllWindows()

out2 = segmentation.segment(image)
out2 = cv2.bitwise_not(out2)
out2 = cv2.bitwise_and(bg,bg,mask=out2)
print(out2)
cv2.imshow("test",out2)
cv2.waitKey(0)
cv2.destroyAllWindows()

out = out1 + out2
print(out)
cv2.imshow("test",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

out_gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
out_gray = out_gray/255
print(out_gray)
cv2.imshow("test",out_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

s_out = segmentation.segment(out)
print(s_out)
cv2.imshow("test",s_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Parameters
# blur = 21
# canny_low = 15
# canny_high = 150
# min_area = 0.0005
# max_area = 0.95
# dilate_iter = 10
# erode_iter = 10
# mask_color = (0.0,0.0,0.0)

# image = cv2.imread('dataset/A/0.jpg')
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(image_gray, canny_low, canny_high)
# cv2.imshow("test",edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# edges = cv2.dilate(edges, None)
# edges = cv2.erode(edges, None)

# contour_info = [(c, cv2.contourArea(c),) 
#                 for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]

# # Get the area of the image as a comparison
# image_area = image.shape[0] * image.shape[1]  
  
# # calculate max and min areas in terms of pixels
# max_area = max_area * image_area
# min_area = min_area * image_area

# mask = np.zeros(edges.shape, dtype = np.uint8)

# for contour in contour_info:
#     # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
#     if contour[1] > min_area and contour[1] < max_area:
#         # Add contour to mask
#         mask = cv2.fillConvexPoly(mask, contour[0], (255))
        
# mask = cv2.dilate(mask, None, iterations=dilate_iter)
# mask = cv2.erode(mask, None, iterations=erode_iter)
# mask = cv2.GaussianBlur(mask, (blur, blur), 0)
# cv2.imshow("test",mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()