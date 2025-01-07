import cv2
import numpy as np
import utlis
from PIL import Image
import pytesseract
import pandas as pd
import os




########################################

path = "OMR.jpeg"
widthImg = 700
heightImg = 1000


########################################

#Reads Img
img = cv2.imread(path)
##PREPROCESSING
#Convert Img to B&W then GaussianBlur and then, Canny Edge Detection.

img = cv2.resize(img,(widthImg,heightImg))
imgBiggestContours= img.copy()
imgContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

#FINDING CONTOURS

contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

#FINDING TARGET RECTANGLE
rectCon = utlis.rectContour(contours)
Largest_contour=utlis.getCornerPoints(rectCon[0]) 
print(Largest_contour.shape)


if Largest_contour.size != 0:
    cv2.drawContours(imgBiggestContours, Largest_contour,-1,(0,0,255),20)
    Largest_contour_points = utlis.reorder(Largest_contour)
    pts1 = np.float32(Largest_contour_points) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE


split_img= utlis.splitBoxes(imgThresh)


cv2.imshow("Original",split_img[0])

imgBlank=np.zeros_like(img)

imageArray=([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpColored,imgThresh])







imgStacked = utlis.stackImages(imageArray,0.5)

#cv2.imshow("Original",imgStacked)
cv2.waitKey(0) 


split_img1 = Image.fromarray(split_img[2])
# Get the original dimensions
width, height = split_img1.size

# Calculate the new dimensions for cropping
new_width = int(4 * width / 5)
new_height = int(30 * height / 31)

# Crop the image (top-left corner as reference)
cropped_img = split_img1.crop((new_width*0.32,new_height*0.039,new_width*1.25,new_height*1.0333))

# Save or display the cropped image
output_dir = r"G:\python project 001\subfolder"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Add a valid file name and extension
cropped_image_path = os.path.join(output_dir, "cropped_image.png")
cropped_img.save(cropped_image_path)
cropped_img.show()  # Opens the cropped image using the default viewer

print(f"Cropped image saved to {cropped_image_path}")


matrix= utlis.image_to_matrix(cropped_image_path)

for row in matrix:
    print(row)
