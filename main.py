
import numpy as np
import cv2

def bubbleSort(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n-1):
    # range(n) also work but outer loop will repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] < arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                final_contours[j], final_contours[j + 1] = final_contours[j + 1], final_contours[j]

path = 'road.jpg'
img_orig = cv2.imread(path)

#image pre-processing
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
kernel = np.ones((2,2), np.uint8)
img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.erode(img, kernel, iterations=1)
cv2.imshow("ss", img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_blue = np.array([60, 35, 140])
# upper_blue = np.array([180, 255, 255])
# mask = cv2.inRange(img, lower_blue, upper_blue)
#warping
# IMAGE_H = img.shape[0]
# IMAGE_W = img.shape[1]

# src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
# Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

# # img = cv2.imread('./test_img.jpg') # Read the test img
# img_w = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
# warped_img = cv2.warpPerspective(img_w, M, (IMAGE_W, IMAGE_H))

# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# twoDimage = img.reshape((-1,3))
# twoDimage = np.float32(twoDimage)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 2
# attempts=20

# ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# result_image = res.reshape((img.shape))
# print(result_image.shape)
# print(img.shape)

#performed canny edge
edges = cv2.Canny(img,50,150)
cv2.imshow("edges",edges)
#---- Next I performed morphological erosion for a rectangular structuring element of kernel size 7 ----
ret, thresh = cv2.threshold(edges, 127, 255, 1)
kernel = np.ones((2, 2),np.uint8)
erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 2)

#---- I then inverted this image and blurred it with a kernel size of 15. The reason for such a huge kernel is to obtain a smooth leaf edge ----
ret, thresh1 = cv2.threshold(erosion, 127, 255, 1)
ret, thresh1_1 = cv2.threshold(thresh1, 127, 255, 1)
cv2.imshow('thresh1_1', thresh1_1)
blur = cv2.blur(thresh1_1, (5, 5))
cv2.imshow('blur', blur)

#---- And then performed morphological erosion to thin the edge. For this I used an ellipse structuring element of kernel size 5 ----
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
final = cv2.morphologyEx(thresh1_1, cv2.MORPH_ERODE, kernel1, iterations = 2)
cv2.imshow('final', final)
contours, hierarchy = cv2.findContours(final, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = len, reverse=True) #sorting the contours through their size


img_to_draw = np.array(img_orig)

#setting the minimum contour size (dependent to img size)
img_size = img_orig.shape[0] * img_orig.shape[1]
min_size = int(img_size*0.00090)

#getting only the contours with large sizes
final_contours = []
for i in contours:
    if len(i)> min_size:
        final_contours.append(i)

#setting the lower and uppwer gray valuues
#for dark gray 
gray_1 = ((np.abs(30-30)+np.abs(30-30)+np.abs(30-30))/3)
gray_2 = np.abs((30 + 30 + 30)/3)-0.5
gray_3_dark = (gray_1+gray_2)/2
#for light gray 
gray_1 = ((np.abs(224-224)+np.abs(224-224)+np.abs(224-224))/3)
gray_2 = np.abs((224 + 224 + 224)/3)-0.5
gray_3_light = (gray_1+gray_2)/2

#drawing the contours
lst_intensities = []
pixel_counter = []
#convert img to hsv
# GRAY_MAX= np.array([180, 18, 230],np.uint8)
# GRAY_MIN = np.array([0, 0, 40],np.uint8)
# cv2.imshow('output2grassy', img_orig)

# imt_test = np.array(img_orig)
# imt_test = cv2.cvtColor(imt_test,cv2.COLOR_BGR2HSV)
# cv2.imshow('output2grasssy', imt_test)
# frame_threshed = cv2.inRange(imt_test, GRAY_MIN, GRAY_MAX)
# cv2.imshow('output2gray', frame_threshed)

for i in range(len(final_contours)):
    convexHull = cv2.convexHull(final_contours[i])
    cimg = np.zeros_like(img_to_draw) #creating a mask
    cv2.drawContours(cimg, [convexHull], -1, color=255, thickness= -1)

    pts = np.where(cimg == 255)
    lst_intensities = [img_orig[pts[0], pts[1]]]
    counter = 0
    #traversing through pixel values of contours
    for x in range(len(lst_intensities)):
        r = int(lst_intensities[0][x][2])
        g = int(lst_intensities[0][x][1])
        b = int(lst_intensities[0][x][0])
        # gray_1 = ((np.abs(r-g)+np.abs(r-b)+np.abs(g-b))/3)
        # gray_2 = np.abs((r + g + b)/3)-0.5
        # gray_3 = (gray_1+gray_2)/2
        #filtering the gray pixels
        # if gray_3 > gray_3_dark and gray_3 < gray_3_light:
        #     counter = counter + 1
        if (r<=125 and r>=40) and (g>=40 and g<=125) and (b<=125 and b>=40):
            counter = counter + 1
    
    #getting how many gray percent is the contour 
    percentage = (counter/len(lst_intensities))*100
    pixel_counter.append(percentage)

#calling bubble sort to sort the contours with highest gray values 
bubbleSort(pixel_counter)
#filterring only contours with 60% above gray areas
new_pixel_counter=[]
for score in pixel_counter:
    if score >=60:
        new_pixel_counter.append(score)

# masking the contours to the original image
final_contours = final_contours[:len(new_pixel_counter)]
masked = np.zeros_like(img_to_draw)

for contour in final_contours:
    print(contour)
    convexHull = cv2.convexHull(contour)
    cv2.drawContours(masked, [convexHull], -1, color=255, thickness= -1)
    pts = np.where(masked == 255)
    masked[pts[0],pts[1]]=img_orig[pts[0], pts[1]]     

cv2.imshow('masked image', masked)
cv2.imshow('original', img_orig)


# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 25  # minimum number of pixels making up a line
# max_line_gap = 10  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img) * 0  # creating a blank to draw lines on

#flood filling 
# edges = edges.astype(np.uint8)
# h,w = img.shape[:2]
# mask = np.zeros((h+2,w+2), np.uint8)
# cv2.floodFill(edges, mask, (250, 250),255)

# result_image = cv2.bitwise_and(img, img, mask = mask)

#hole filling
# im_floodfill_inv = cv2.bitwise_not(result_image)
# im_out = img | im_floodfill_inv

# edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

# transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

# transparent[:,:,0:3] = img
# transparent[:, :,3] = edges
# cv2.imshow("masked",transparent)
# cv2.imshow("edge",edges)
# cv2.imshow("warped",warped_img)



cv2.waitKey()
cv2.destroyAllWindows()