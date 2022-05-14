# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:39:57 2022

@author: 6B01
"""
from cv2 import fastNlMeansDenoising, imwrite, imread
from programs.image_process import to_digits, select, get_intersect

def split_image(image):
    
    croppedImages = []
    
    # 取得裁切點
    x_point_arr, y_point_arr = select(get_intersect(image))
    
    print('圖像降噪中...')
    # 圖像降噪
    denoisedImage = fastNlMeansDenoising(image, h=10)
       
    # 取得數字影像
    digitImage = to_digits(denoisedImage)
        
    print('裁切圖片中...')
    # 裁切
    for i in range(1, len(y_point_arr)):
        temp = []
        for j in range(1, len(x_point_arr)):
            x1, x2, y1, y2 = map(int, (x_point_arr[j-1], x_point_arr[j],\
                                       y_point_arr[i-1], y_point_arr[i]))
            cropImage = digitImage[y1:y2, x1:x2]
            imwrite(r"content\croppedImages\{0}_{1}.png".format(i, j), cropImage)
            temp.append(cropImage)
        croppedImages.append(temp)

    return croppedImages