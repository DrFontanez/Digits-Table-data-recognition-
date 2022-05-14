# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:00:22 2022

@author: 6B01
"""
'''圖像切割'''

from cv2 import cvtColor, COLOR_BGR2GRAY, getStructuringElement, MORPH_RECT, \
    morphologyEx, MORPH_OPEN, adaptiveThreshold, \
    ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, add, imwrite, subtract, \
    MORPH_CLOSE, bitwise_and, dilate
from numpy import where, sort

def to_lines(img):
    
    # 轉灰階
    gray = cvtColor(img, COLOR_BGR2GRAY)
    imwrite(r"content\excelImages\_1gray.png", gray)
    
    # 影像二值化
    binary = adaptiveThreshold(
        ~gray,
        255, 
        ADAPTIVE_THRESH_GAUSSIAN_C, 
        THRESH_BINARY, 
        35,
        -5
    )
    imwrite(r"content\excelImages\_2binary.png", binary)
    
    rows, cols = binary.shape # 圖片長,寬
    
    print('偵測線條中...')
    '''進行水平開運算，取得橫線'''
    kernel = getStructuringElement(MORPH_RECT, (cols // 15, 1))
    dilated_col = morphologyEx(binary, MORPH_OPEN, kernel) # 影像處理結果(橫線)
    imwrite(r"content\excelImages\_Ocol.png", dilated_col)
    
    '''進行垂直開運算，取得直線'''
    kernel = getStructuringElement(MORPH_RECT, (1, rows // 5))
    dilated_row = morphologyEx(binary, MORPH_OPEN, kernel) # 影像處理結果(直線)
    imwrite(r"content\excelImages\_Orow.png", dilated_row)
    
    return dilated_row, dilated_col, binary

def get_intersect(img):
    
    # 取得直線影像與橫線影像
    dilated_row, dilated_col = to_lines(img)[:2]
    
    '''取得直線與橫線同有的點'''
    bitwiseAnd = bitwise_and(dilated_col, dilated_row)
    dilation = dilate(bitwiseAnd, (5, 5))
    imwrite(r"content\excelImages\_4excel_bitwise_and.png", dilation)

    return where(bitwiseAnd > 0)

def to_digits(img):
    
    # 取得直線影像與橫線影像
    dilated_row, dilated_col, binary = to_lines(img)
    
    print('處理文字影像中...')
    
    '''直線橫線圖像相加，取得表格輪廓'''
    excel = add(dilated_col, dilated_row)
    imwrite(r"content\excelImages\_3excel_add.png", excel)
    

    '''二值化圖像與表格輪廓相減，移除表格輪廓'''
    digits = subtract(binary, excel)
    imwrite(r"content\excelImages\_5excel_subtract.png", digits)

    '''進行開運算，去除雜訊'''
    kernelSize = (2, 2)
    new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    morphOpen = morphologyEx(digits, MORPH_OPEN, new_kernel)
    imwrite(r"content\excelImages\_6excel_morphologyEx.png", morphOpen)
    
    # '''進行閉運算，修復斷裂文字'''
    # kernelSize = (3, 3)
    # new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    # morphClose = morphologyEx(morphOpen, MORPH_CLOSE, new_kernel)
    # imwrite(r"content\excelImages\_7excel_morphologyExCLOSE.png", morphClose)
    
    return morphOpen

def select(coordianate):
    ys, xs = coordianate

    y_point_arr = []
    x_point_arr = []

    print('取得裁切點中...')    

    '''X軸'''
    
    # 排序座標數值，將接近值合併為一個座標
    sort_x_point = sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 8:
            x_point_arr.append(sort_x_point[i])
        i = i + 1
    x_point_arr.append(sort_x_point[i])

    # 取得最大間距，合併低於一半的值
    res=0
    for i in range(1, len(x_point_arr)):
        if x_point_arr[i]-x_point_arr[i-1] > res:
            res = (x_point_arr[i]-x_point_arr[i-1])/2
    
    j=0
    for i in range(len(x_point_arr)-1):
        if abs(x_point_arr[i-j]-x_point_arr[i-j+1]) < res:
            x_point_arr[i-j] = x_point_arr[i-j+1]
            del x_point_arr[i-j+1]
            j+=1
    
    '''Y軸'''
    
    # 排序座標數值，將接近值合併為一個座標
    sort_y_point = sort(ys)
    for i in range(len(sort_y_point) - 1):
        if sort_y_point[i + 1] - sort_y_point[i] > 8:
            y_point_arr.append(sort_y_point[i])
        i = i + 1
    y_point_arr.append(sort_y_point[i])
        
    # 取得最大間距，合併低於一半的值
    res=0
    for i in range(1, len(y_point_arr)):
        if y_point_arr[i]-y_point_arr[i-1] > res:
            res = (y_point_arr[i]-y_point_arr[i-1])/2
    
    j=0
    for i in range(len(y_point_arr)-1):
        if abs(y_point_arr[i-j]-y_point_arr[i-j+1]) < res:
            y_point_arr[i-j] = y_point_arr[i-j+1]
            del y_point_arr[i-j+1]
            j+=1

    return sorted(x_point_arr), sorted(y_point_arr)