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

    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # print('Original Dimensions : ',img_array.shape)
    
    scale_percent = 80 # percent of original size
    width = int(img_array.shape[1] * scale_percent / 100)
    height = int(img_array.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)
    
    def crop(x, y, w, h, img_type):
        # if img_type == 'ref':
        #     img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,255),2)
        #     x=x+15
        #     y=y+1
        #     crop_img = img_array[y:y+h, x:x+w]
        # elif img_type == 'device':
        #     img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
        #     x=x+1
        #     y=y+1
        #     crop_img = img_array[y:y+h, x:x+w]
        # elif img_type == 'lot':
        #     img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
        #     x=x+1
        #     y=y+1
        #     crop_img = img_array[y:y+h, x:x+w]
        # elif img_type == 'qty':
        #     img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,55,0),2)
        #     x=x+1
        #     y=y+1
        #     crop_img = img_array[y:y+h, x:x+w]
        # elif img_type == 'symbol':
        #     img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,205,100),2)
        #     x=x+1
        #     y=y+1
        #     crop_img = img_array[y:y+h, x:x+w]
        # else:
        #     print('Correct Type!')
        if img_type == 'lot':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,0,255),2)
        elif img_type == 'device':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
        elif img_type == 'ref':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
        elif img_type == 'qty':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,55,0),2)
        elif img_type == 'symbol':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,205,100),2)
        else:
            print('Correct Type!')
        boxed_img = cv2.imwrite("boxed_img{}.jpg" .format(i), img)
        x=x+8
        y=y+8
        crop_img = img_array[y:y+h, x:x+w]
        return crop_img, img, boxed_img

    cropped_lot_img = crop(180,200,220,70, img_type='lot')
    cropped_device_img = crop(310,100,280,50, img_type='device')
    cropped_ref_img = crop(220,320,380,160, img_type='ref')
    cropped_qty_img = crop(1080,220,980,110, img_type='qty')
    cropped_symbol_img = crop(980,220,920,110, img_type='symbol')

    cv2.imwrite("cropped_ref_img.jpg", cropped_ref_img[0])
    cv2.imwrite("Image_output\device\cropped_device_img.jpg",cropped_device_img)
    cv2.imwrite("Image_output\lot\cropped_lot_img.jpg",cropped_lot_img)
    cv2.imwrite("Image_output\qty\cropped_qty_img.jpg", cropped_qty_img)
    cv2.imwrite("Image_output\symbols\cropped_symbol_img.jpg", cropped_symbol_img)
    # cv2.imwrite("croped_img/Cropped_Image{}.jpg" .format(i), cropped_image)
    # cv2.imshow('output_img.jpg', img_arr)
    # plt.show()



# img = cv2.imread('images/result_Page_1.jpg')
# print(img.shape) # Print image shape
# cv2.imshow("original", img)
 
# # Cropping an image
# cropped_image = img[100:210, 410:730]
 
# # Display cropped image
# cv2.imshow("cropped", cropped_image)
 
# # Save the cropped image
# for i in range(10):
#     print(i)
#     cv2.imwrite("croped_img/Device_name/Cropped_Image{}.jpg" .format(i), cropped_image)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
 
# img1 = cv2.imread('sample.jpg')
# img2 = cv2.imread('sample2.jpg')
 
# orb = cv2.ORB_create(nfeatures=500)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
 
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
 
# matches = sorted(matches, key=lambda x: x.distance)
 
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
 
# cv2.imshow(match_img)
# cv2.waitKey()