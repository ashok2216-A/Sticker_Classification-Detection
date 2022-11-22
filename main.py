# Import packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


datadir = 'images'
for i in range(len(os.listdir(datadir))):
    # print(i)
    filename = 'input_img{}.jpg' .format(i)
    path = os.path.join(datadir,filename)
    print(path)
    img_path = path
    img_array = cv2.imread(img_path)
    ref_axis, ref_box = (210,200),(100+380,210+110)
    device_axis, device_box = ((370,180),(270+380,110+8))
    lot_axis, lot_box = ((220,320),(100+380,210+160))
    qty_axis, qty_box = ((1080,220),(280+980,210+110))
    symbol_axis1, symbols_box1 = ((1060,220),(320+920,10+110))
    symbol_axis2, symbols_box2 = ((140,420),(300+430,210+420))

    img=cv2.rectangle(img_array,ref_axis, ref_box,(0,0,255),2)
    img=cv2.rectangle(img_array,device_axis, device_box,(0,255,255),2)
    lot_img=cv2.rectangle(img_array,lot_axis,lot_box,(255,0,0),2)
    qty_img=cv2.rectangle(img_array,qty_axis,qty_box,(0,255,0),2)
    symbol_img1=cv2.rectangle(img_array,symbol_axis1,symbols_box1,(255,55,0),2)
    symbol_img2=cv2.rectangle(img_array,symbol_axis2,symbols_box2,(0,205,100),2)

    cropped_image = img_array[100:210, 410:730]
    cv2.imwrite("img.jpg" .format(i), img)
    # cv2.imwrite("croped_img/Cropped_Image{}.jpg" .format(i), cropped_image)
    # cv2.imshow('output_img.jpg', img_arr)
    # plt.show()



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


import cv2
 
img1 = cv2.imread('sample.jpg')
img2 = cv2.imread('sample2.jpg')
 
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
 
matches = sorted(matches, key=lambda x: x.distance)
 
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
 
cv2.imshow(match_img)
cv2.waitKey()