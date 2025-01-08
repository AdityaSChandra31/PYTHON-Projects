import cv2
import numpy as np
import utlis
from PIL import Image
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



imgBlank=np.zeros_like(img)

imageArray=([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpColored,imgThresh])



imgStacked = utlis.stackImages(imageArray,0.5)


output_dir = r"G:\PYTHON-Projects\OMR Reader\subfolder"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

pixels=[]

# Iterate over each image in split_img
for i, img in enumerate(split_img):
    # Convert the image to a PIL image to crop
    split_img1 = Image.fromarray(img)
    
    # Get the original dimensions
    width, height = split_img1.size
    
    # Calculate the new dimensions for cropping
    new_width = int(4 * width / 5)
    new_height = int(30 * height / 31)
    
    # Crop the image (top-left corner as reference)
    cropped_img = split_img1.crop((new_width * 0.32, new_height * 0.039, new_width * 1.25, new_height * 1.0333))
    
    # Save the cropped image
    cropped_image_path = os.path.join(output_dir, f"cropped_image_{i}.png")
    cropped_img.save(cropped_image_path)
    print(f"Cropped image {i} saved to {cropped_image_path}")
    
    # Process cropped image for further slicing
    crop_img = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)
    boxes = utlis.sliceImage(crop_img)
    
    for box in boxes:
        pixel_sum = utlis.sum_pixel_values(box)  # Get the sum of pixel values for each box
        pixels.append(pixel_sum)         



utlis.process_pixels(pixels)
