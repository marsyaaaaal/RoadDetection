
import numpy as np
import cv2
import argparse
from functions import *

# USAGE: python road.py -i imagefilename.jpg

# NOTE: All kernel sizes are adjustable dependent to the results of testing
# but this default sizes are the ones we recommend during our testin
                


# Getting the image filename on the commandline      
ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-i","--image", required=True,
   help="image filename", type=check_input)
args = vars(ap.parse_args())
path = args['image'] # filename of image
img_orig = cv2.imread(path)
#convert image to 1280, 720
img_orig = cv2.resize(img_orig, (1280,720))

# img=cv2.ximgproc.anisotropicDiffusion(img_orig, 0.25, 1,1)

#image pre-processing, converting to grayscale, histogram and applying smoothing using gaussian and erosion
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

#pre-process 
img, v, kernel, min_value = pre_process(img, img_orig)

#performing canny edge detection
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = lower * 3
edges = cv2.Canny(img,lower,upper,3) # default 75, 200

#thresholdings
final_contours = thresholding(edges, img_orig, kernel)
#pixel elimination
mask_contours = gray_pixel_elimination(final_contours, min_value, img_orig)
#merging and post process
bg, roi_road = post_process(img_orig, mask_contours)

# blending the red area road surface to the original image
alpha = 0.25
road_surface = cv2.addWeighted(img_orig, 1-alpha, bg, alpha, 0)

winname = "Final road surface"
cv2.imshow(winname, road_surface)
cv2.waitKey()
cv2.destroyAllWindows()