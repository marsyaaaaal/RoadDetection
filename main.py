
import numpy as np
import cv2

#NOTE: All kernel sizes are adjustable dependent to the results of testing
# but this default sizes are the ones we recommend during our testin

# sorting algorithm for sorting of contour sizes
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

path = 'road18.jpg' # filename/path of image
img_orig = cv2.imread(path)
cv2.imshow("Original Inputted Image", img_orig)

#image pre-processing, converting to grayscale, histogram and applying smoothing using gaussian and erosion
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img) #performing histogram equalization to equalize the distribution of pixels
kernel = np.ones((2,2), np.uint8) # small kernel size only for erosion 
img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.erode(img, kernel, iterations=1)
cv2.imshow("Pre-processed image", img)

#performing canny edge detection
edges = cv2.Canny(img,75,200,3) # default 75, 180
cv2.imshow("Canny edges",edges)

# Inverting the edges image then running erosion to make the edges thicker and eliminate false edges
ret, inverted = cv2.threshold(edges, 127, 255, 1)
erosion = cv2.morphologyEx(inverted, cv2.MORPH_ERODE, kernel, iterations = 2)
cv2.imshow('Inverted and Erode', erosion)
#Applying final erosion using getStructuralElement of ellipse kernel size of 3,3
#Using MORPH_ELLIPSE makes the edges/lines ellipse shaped rather than use MORPH_RECT since ellipses are more flexible than rectangles
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
final = cv2.morphologyEx(erosion, cv2.MORPH_ERODE, kernel1, iterations = 2)
cv2.imshow('Processed', final)

#Finding contours/shapes/descriptor from the final processed image
contours, hierarchy = cv2.findContours(final, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = len, reverse=True) #sorting the contours with respect to their their size


#setting the minimum contour/descriptor size (dependent to image area)
img_size = img_orig.shape[0] * img_orig.shape[1]
min_size = int(img_size*0.00090) # adjustable size, depends on testing state to adjust the value

#getting only the contours with large sizes grater than the min_size initialized above
final_contours = []
for i in contours:
    if len(i)> min_size:
        final_contours.append(i)

# Here we determine if the image is light or dark, we used HSV color space and calulated the mean of V
# We copied the original image to another variable and converted it to HSV
imt_test = np.array(img_orig)
imt_test = cv2.cvtColor(imt_test,cv2.COLOR_BGR2HSV)
average_value = np.mean(imt_test[:,1,2])

# We assumed that if the image is dark (average V is lower than or equal 40) the pixels of the image (especially the road) will be lowered to
# therefore we must, lower also the minimum value to 0, otherwise if the image is light (average V is greater than 40) the pixels
# of the image will have a high values so we will adjust the min value to -50
if average_value <=40: 
    min_value = 0
else:
    min_value = -50

contour_pixels = [] # varaible storage for pixels of contours
gray_pixel_counter = [] # varaible storage for number of gray pixels of each contours

# Here we loop through all candidate contours/descriptors for road surface and then count the pixels inside each 
# contours within the gray space.
for contour in final_contours:
    cimg = np.zeros_like(img_orig) #creating a mask for each contour
    cv2.drawContours(cimg, [contour], -1, color=255, thickness= -1) # drawing the contour inside the empty mask and coloring them 255(white)
    pts = np.where(cimg == 255)# extracting points from the mask that has a value of 255(white)
    contour_pixels = [img_orig[pts[0], pts[1]]] # since we got the points of the contour, we get the pixels from the image using the extracted points  
    counter = 0
    
    #traversing through pixel values of contours to know if the pixel is within the gray level
    for x in range(len(contour_pixels[0])):
        r = int(contour_pixels[0][x][2]) # red value of pixel x
        g = int(contour_pixels[0][x][1]) # green value of pixel x
        b = int(contour_pixels[0][x][0]) # blue value of pixel x
  
        # if one of the rgb values is too high(higher than 200 ) or low, dont execute 
        if ( r < 200 and g < 200 and b < 200) and ( r > 20 and g > 20 and b > 20):
            if (r-g-b < min_value):
                counter = counter + 1 # iterate counter
                  
    #getting how many gray percent is within the contour 
    percentage = (counter/len(contour_pixels[0]))*100
    gray_pixel_counter.append(percentage)

#Sort the pixels
bubbleSort(gray_pixel_counter)

#eliminating contours that is lower than 60% gray pixels  
final_gray_contour_scores=[] # new variable storage of contours that has gray pixels higher than or equal to 60%
for score in gray_pixel_counter:
    if score >=60:
        final_gray_contour_scores.append(score)
        
# Getting only contours that are remained from filtering above
final_contours = final_contours[:len(final_gray_contour_scores)]
bg = img_orig.copy() # making a copy of original image
masked = np.zeros_like(img_orig) # making a mask for drawing the contours
# Drawing the final contours to the empty image
for contour in final_contours:
    cv2.drawContours(masked, [contour], -1, color=255, thickness= 2)
    pts = np.where(masked == 255)
    masked[pts[0],pts[1]]= (255,255,255)

#Merging nearby Contours to eliminate small spaces between them
masked = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
# using the MORPH_CLOSE to merge the nearby contours using getStructuringElement kernel ellipse morph and size of 20,20
merged_contours = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))

# We assume that nearby contours are parts of the road that are separated of lines/edges
# Find contours in merged_contours after closing the gaps 
# So for example 2 contours are merged then will be classified as one contour
contours, hier = cv2.findContours(merged_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#compute for the 5% of img size as minimum contour area
img_size = img_orig.shape[0] * img_orig.shape[1]
min_size = int(img_size*0.05)

# Drawing and blending the final contours 
for contour in contours:
    area = cv2.contourArea(contour) # getting the area of each contour
    # contours lower than 5% of img size are ignored
    if area < min_size:
        continue
    # draw the contours that are larger than 5% of img size 
    # convex = cv2.convexHull(contour)
    cv2.drawContours(bg, [contour], -1, (0,0,255), thickness= -1)
    
# blending the red area road surface to the original image
alpha = 0.25
final_ouput = cv2.addWeighted(img_orig, 1-alpha, bg, alpha, 0)

cv2.imshow('Final Contours', merged_contours)
cv2.imshow('Final road surface', final_ouput)
cv2.waitKey()
cv2.destroyAllWindows()