# Import packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


datadir = 'images'
for i in range(len(os.listdir(datadir))):
    # print(i)
    filename = 'input_img{}.jpg' .format(i)
    path = os.path.join(datadir,filename)
    print(path)
    img_path = path
    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    scale_percent = 80 # percent of original size
    width = int(img_array.shape[1] * scale_percent / 100)
    height = int(img_array.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)
    def crop(x, y, w, h, img_type):
        if img_type == 'ref':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,0,0),2)
        elif img_type == 'lot':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,0,255),2)
        elif img_type == 'device':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
        elif img_type == 'qty':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,255,10),2)
        elif img_type == 'symbol1':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,55,100),2)
        elif img_type == 'symbol2':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            print('Correct Type!')
        boxed_img = cv2.imwrite("blp_img/boxed_img{}.jpg" .format(i), img)
        x=x+0
        y=y+0
        crop_img = img_array[y:y+h, x:x+w]
        return crop_img, img, boxed_img
    cropped_ref_img = crop(190,150,200,60, img_type='ref')
    cropped_lot_img = crop(180,220,220,70, img_type='lot')
    cropped_device_img = crop(310,100,280,50, img_type='device')
    cropped_symbol1_img = crop(120,320,680,140, img_type='symbol1')
    cropped_qty_img = crop(820,160,280,120, img_type='qty')
    cropped_symbol2_img = crop(800,80,200,80, img_type='symbol2')

    cv2.imwrite("Image_output\Ref\cropped_Ref_img{}.jpg" .format(i), cropped_ref_img[0])
    cv2.imwrite("Image_output\device\cropped_device_img{}.jpg" .format(i), cropped_device_img[0])
    cv2.imwrite("Image_output\lot\cropped_lot_img{}.jpg" .format(i), cropped_lot_img[0])
    cv2.imwrite("Image_output\qty\cropped_qty_img{}.jpg" .format(i), cropped_qty_img[0])
    cv2.imwrite("Image_output\symbols\cropped_symbol_img{}.jpg" .format(i), cropped_symbol1_img[0])
    cv2.imwrite("Image_output\symbols\cropped_symbol_img{}.jpg" .format(i), cropped_symbol2_img[0])
    # cv2.imshow('output_img.jpg', img_arr)
    # plt.show()
    dir_file = ['Ref', 'lot', 'qty', 'device']
    # image_path = 'Image_output'+'/'+dir_file[i]+'/cropped_'+dir_file[i]+'_img{}.jpg'
    # print(image_path)
    for ls in range(len(dir_file)):
        for bs in range(4):
            img = cv2.imread('Image_output'+'/'+dir_file[ls]+'/cropped_'+dir_file[bs]+'_img{}.jpg' .format(bs))
            # print(img)
        # print(img)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\AppData\Local\Tesseract-OCR\tesseract.exe'

    text = pytesseract.image_to_string(img)
    print(text)
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
# dir_file = ['Ref', 'lot', 'qty', 'device']
# for i in range(len(dir_file)):
#     image_path = 'Image_output'+'/'+dir_file[i]+'/cropped_'+dir_file[i]+'_img{}.jpg'
#     print(image_path)
import cv2
dir_file = ['Ref', 'lot', 'qty', 'device']
    # image_path = 'Image_output'+'/'+dir_file[i]+'/cropped_'+dir_file[i]+'_img{}.jpg'
    # print(image_path)
for ls in range(len(dir_file)):
    img = 'Image_output'+'/'+dir_file[ls]+'/cropped_'+dir_file[ls]+'_img{}.jpg' .format(ls)
    print(img)