
import numpy as np
import cv2
import argparse

def check_input(img):
    if cv2.haveImageReader(img):
        return img
    raise argparse.ArgumentTypeError("Invalid File, either missing or invalid file format",img)
    

def pre_process(img, img_orig):
    
    # Here we determine if the image is light or dark, we used HSV color space and calulated the mean of V
    # We copied the original image to another variable and converted it to HSV
    imt_test = np.array(img_orig)
    imt_test = cv2.cvtColor(imt_test,cv2.COLOR_BGR2HSV)
    average_value = np.mean(imt_test[:,1,2])
   
    img = cv2.equalizeHist(img) #performing histogram equalization to equalize the distribution of pixels
    # We assumed that if the image is dark (average V is lower than or equal 40) the pixels of the image (especially the road) will be lowered to
    # therefore we must, lower also the minimum value to 0, otherwise if the image is light (average V is greater than 40) the pixels
    # of the image will have a high values so we will adjust the min value to -50
    if average_value <=40: 
        min_value = 0
    elif average_value >=190:
        # if the image is too bright, adjust the brightness
        brightness = -50
        contrast = -50
        img = np.int16(img_orig)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img_orig = np.copy(img)
        min_value = -40 
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img) #performing histogram equalization to equalize the distribution of pixels
    else:
        min_value = -50

    kernel = np.ones((2,2), np.uint8) # small kernel size only for erosion 
    img = cv2.GaussianBlur(img,(3,3),0)
    v = np.median(img)

    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations = 1)

    
    return img, v, kernel, min_value

def thresholding(edges, img_orig, kernel):
    # Inverting the edges image then running erosion to make the edges thicker and eliminate false edges
    _, inverted = cv2.threshold(edges, 127, 255, 1)
    erosion = cv2.morphologyEx(inverted, cv2.MORPH_ERODE, kernel, iterations = 2)
    #Applying final erosion using getStructuralElement of ellipse kernel size of 3,3
    #Using MORPH_ELLIPSE makes the edges/lines ellipse shaped rather than use MORPH_RECT since ellipses are more flexible than rectangles
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    final = cv2.morphologyEx(erosion, cv2.MORPH_ERODE, kernel1, iterations = 2)
    #Finding contours/shapes/descriptor from the final processed image
    contours, _ = cv2.findContours(final, 
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

    return final_contours

def gray_pixel_elimination(final_contours, min_value, img_orig):
    contour_pixels = [] # varaible storage for pixels of contours
    # Here we loop through all candidate contours for road surface and then count the pixels inside each 
    # contours within the gray space.
    #creating mask for contours to make a new masked image with all the pixels that are considered gray
    mask_contours = np.zeros_like(img_orig)
    for contour in final_contours:
        cimg = np.zeros_like(img_orig) #creating a mask for each contour
        cv2.drawContours(cimg, [contour], -1, color=255, thickness= -1) # drawing the contour inside the empty mask and coloring them 255(white)
        pts = np.where(cimg == 255)# extracting points from the mask that has a value of 255(white)
        contour_pixels = [img_orig[pts[0], pts[1]]] # since we got the points of the contour, we get the pixels from the image using the extracted points  
        #traversing through pixel values of contours to know if the pixel is within the gray level
        for x in range(len(contour_pixels[0])):
            r = int(contour_pixels[0][x][2]) # red value of pixel x
            g = int(contour_pixels[0][x][1]) # green value of pixel x
            b = int(contour_pixels[0][x][0]) # blue value of pixel x
            # if one of the rgb values is too high(higher than 200 ) or low, dont execute 
            if ( r < 200 and g < 200 and b < 200) and ( r > 20 and g > 20 and b > 20):
                if (r-g-b < min_value):
                    # counter = counter + 1 # iterate counter
                    mask_contours[pts[0][x],pts[1][x]] = (255,255,255)
                else:
                    mask_contours[pts[0][x],pts[1][x]] = (0,0,0)
            else:
                mask_contours[pts[0][x],pts[1][x]] = (0,0,0)

    return mask_contours

def post_process(img_orig, mask_contours):


    mask_contours = cv2.cvtColor(mask_contours,cv2.COLOR_BGR2GRAY)
    conts, _ = cv2.findContours(mask_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_mask = np.zeros_like(img_orig)
    #compute for the 5% of img size as minimum contour area
    img_size = img_orig.shape[0] * img_orig.shape[1]
    min_size = int(img_size*0.05)

    for c in conts:
        area = cv2.contourArea(c) # getting the area of each contour
        # contours lower than 10% of img size are ignored
        if area < min_size:
            continue
        # draw the contours that are larger than 10% of img size 
        cv2.drawContours(final_mask, [c], -1, (0,0,255), thickness= -1)

    bg = img_orig.copy() # making a copy of original image

    #Merging nearby Contours to eliminate small spaces between them
    masked = cv2.cvtColor(final_mask,cv2.COLOR_BGR2GRAY)

    # using the MORPH_CLOSE to merge the nearby contours using getStructuringElement kernel ellipse morph and size of 20,20
    merged_contours = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))

    # We assume that nearby contours are parts of the road that are separated of lines/edges
    # Find contours in merged_contours after closing the gaps 
    # So for example 2 contours are merged then will be classified as one contour
    contours, _ = cv2.findContours(merged_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_mask = np.zeros_like(img_orig)
    # Drawing and blending the final contours  and the final elimination
    for contour in contours:
        area = cv2.contourArea(contour) # getting the area of each contour
        # contours lower than 5% of img size are ignored
        if area < min_size:
            continue
        # draw the contours that are larger than 5% of img size 
        cv2.drawContours(bg, [contour], -1, (0,0,255), thickness= -1)
        cv2.drawContours(final_mask, [contour], -1, (255,255,255), thickness= -1)
    
    final_mask = cv2.cvtColor(final_mask,cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return bg, cnts