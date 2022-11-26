# Import packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from openpyxl import Workbook


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
    cv2.imwrite("Image_output\symbols1\cropped_symbol_img{}.jpg" .format(i), cropped_symbol1_img[0])
    cv2.imwrite("Image_output\symbols2\cropped_symbol_img{}.jpg" .format(i), cropped_symbol2_img[0])
    # cv2.imshow('output_img.jpg', img_arr)
    # plt.show()
    dir_file = ['Ref', 'lot', 'qty', 'device']
    # image_path = 'Image_output'+'/'+dir_file[i]+'/cropped_'+dir_file[i]+'_img{}.jpg'
    # print(image_path)
    # for ls in range(len(dir_file)):
    #     for bs in range(4):
    ref_img = cv2.imread('Image_output/Ref/cropped_Ref_img{}.jpg' .format(i))
    lot_img = cv2.imread('Image_output/lot/cropped_lot_img{}.jpg' .format(i))
    qty_img = cv2.imread('Image_output/qty/cropped_qty_img{}.jpg' .format(i))
    device_img = cv2.imread('Image_output/device/cropped_device_img{}.jpg' .format(i))
            # print(img)
        # print(img)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\AppData\Local\Tesseract-OCR\tesseract.exe'

    ref_text = pytesseract.image_to_string(ref_img)
    print(ref_text)
    lot_text = pytesseract.image_to_string(lot_img)
    print(lot_text)
    qty_text = pytesseract.image_to_string(qty_img)
    print(qty_text)
    device_text = pytesseract.image_to_string(device_img)
    print(device_text)

    fruits = ['Apple','Peach','Cherry','Watermelon']
    sales = [100,200,300,400]

    row_start = 3  #start below the header row 2
    col_start = 2  #starts from column B
    for i in range(0,len(fruits)):
        ws1.cell(row_start+i, col_start).value = fruits[i]
        ws1.cell(row_start+i, col_start+1).value = sales[i]

import cv2
dir_file = ['Ref', 'lot', 'qty', 'device']
    # image_path = 'Image_output'+'/'+dir_file[i]+'/cropped_'+dir_file[i]+'_img{}.jpg'
    # print(image_path)
for ls in range(len(dir_file)):
    img = 'Image_output'+'/'+dir_file[ls]+'/cropped_'+dir_file[ls]+'_img{}.jpg' .format(ls)
    print(img)