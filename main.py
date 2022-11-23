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
    # ref_axis, ref_box = 
    # device_axis, device_box = ((370,180),(270+380,110+8))
    # lot_axis, lot_box = ((220,320),(100+380,210+160))
    # qty_axis, qty_box = ((1080,220),(280+980,210+110))
    # symbol_axis1, symbols_box1 = ((1060,220),(320+920,10+110))
    # symbol_axis2, symbols_box2 = ((140,420),(300+430,210+420))
    def crop(x, y, w, h,img_type):
        if img_type == 'ref':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,255),2)
            x=x+1
            y=y+1
            crop_img = img_array[y:y+h, x:x+w]
        elif img_type == 'device':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
            x=x+1
            y=y+1
            crop_img = img_array[y:y+h, x:x+w]
        elif img_type == 'lot':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
            x=x+1
            y=y+1
            crop_img = img_array[y:y+h, x:x+w]
        elif img_type == 'qty':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,55,0),2)
            x=x+1
            y=y+1
            crop_img = img_array[y:y+h, x:x+w]
        elif img_type == 'symbol':
            img=cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,205,100),2)
            x=x+1
            y=y+1
            crop_img = img_array[y:y+h, x:x+w]
        else:
            print('Correct Type!')
        
        # x=x+1
        # y=y+1
        # crop_img = img[y:y+h, x:x+w]
        return crop_img
    # cropped_img = crop(210,200,380,110)

    cropped_img = crop(210,200,380,110, img_type='ref')

    # cropped_image = img_array[100:210, 410:730]
    cv2.imwrite("img.jpg" .format(i), cropped_img)
    # cv2.imwrite("bimg.jpg" .format(i), img)