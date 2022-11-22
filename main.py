# Import packages
import os
import cv2
import numpy as np

datadir = 'images'
# print(len(os.listdir(datadir)))
for i in range(len(os.listdir(datadir))):
    print(i)
    path = os.path.join(datadir,)
    for img in os.listdir(datadir):
        img_arr = cv2.imread(os.path.join(datadir,img))




img = cv2.imread('images/result_Page_1.jpg')
print(img.shape) # Print image shape
cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[100:210, 410:730]
 
# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
for i in range(10):
    print(i)
    cv2.imwrite("croped_img/Device_name/Cropped_Image{}.jpg" .format(i), cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()