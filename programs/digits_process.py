# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:44:19 2022

@author: 6B01
"""

from cv2 import getStructuringElement, MORPH_RECT, morphologyEx, MORPH_OPEN, \
    dilate, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect, \
    rectangle, threshold, THRESH_BINARY_INV, copyMakeBorder, BORDER_CONSTANT, \
    resize, INTER_NEAREST, cvtColor, COLOR_GRAY2BGR, imwrite
from numpy import ceil

from programs.image_crop import split_image

def get_digits(image):
    
    boundboxes = []
    
    image = image.copy()
    
    kernelSize = (2, 2)
    new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    final = morphologyEx(image, MORPH_OPEN, new_kernel)

    dilation = dilate(final, (5, 5))
    
    (contours, _) = findContours(dilation, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, width, height = boundingRect(cnt)
        
        boundboxes.append(((x, y), (x+width, y+height)))
        rectangle(image,(x, y),(x+width, y+height),(255, 255, 255))
        
    return sorted(boundboxes)

def digit_image(image):
    
    croppedImages = []
    
    boundboxes = get_digits(image)
    
    binary = threshold(image, 230, 255, THRESH_BINARY_INV)[1]
    
    for b in boundboxes:
        
        x, y, bwidth, bheight = b[0][0], b[0][1], b[1][0], b[1][1]
        
        cropImage = binary[y:bheight, x:bwidth]
        
        croppedImages.append(cropImage)
        
    return croppedImages

def to_revision(image):
    
    height, width = image.shape[:2]
    
    # 計算長寬差額，用於補充較低者
    diff = abs(height-width)/2
    extraheight, extrawidth = (0, 0)
    
    if height > width:
        extrawidth = int(ceil(diff))
        if extrawidth > diff:
            extraheight += 1
    
    if width > height:
        extraheight = int(ceil(diff))
        if extraheight > diff:
            extrawidth += 1
    
    # 補充成正方形
    border = copyMakeBorder(
        image,
        top=extraheight,
        bottom=extraheight,
        left=extrawidth,
        right=extrawidth,
        borderType=BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    borderlength = max(border.shape[:2])
    
    # 添加空白外框
    bordersize = int(borderlength*(4/borderlength))
    border = copyMakeBorder(
        border,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    
    border = resize(border, (224, 224), interpolation=INTER_NEAREST)

    return cvtColor(border, COLOR_GRAY2BGR)

def split_digits(image):
    
    cp = split_image(image)
    
    digits = []
    print('裁切文字&調整文字影像中...')
    for i in range(len(cp)):
        temp=[]
        for j in range(len(cp[i])):
            digtemp=[]
            croppedImage = cp[i][j]
            croppedDigits = digit_image(croppedImage)
            for k in range(len(croppedDigits)):
                digit = croppedDigits[k]
                
                final=to_revision(digit)
                imwrite(r"content\digitImages\{0}_{1}_{2}.png".format(i, j, k), final)
                digtemp.append(final)
            temp.append(digtemp)
        digits.append(temp)
    
    return digits