'''更改影像尺寸'''
from cv2 import INTER_AREA, INTER_CUBIC, resize, imwrite
def to_size(image, size):
    height, width = image.shape[:2]
    scale = height / size
    dim = (int(width/scale), size)
    
    resizeMode = None
    if scale < 1:
        resizeMode = INTER_AREA
    if scale > 1:
        resizeMode = INTER_CUBIC

    if resizeMode:
        image = resize(image, dim, interpolation=resizeMode)

    return image;

'''影像二質化(關係演算法)'''
from cv2 import adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV
def to_adaptivebinary(img, ref=11):
    
    binaryImage = adaptiveThreshold(
        img, 
        255, 
        ADAPTIVE_THRESH_MEAN_C, 
        THRESH_BINARY_INV, 
        ref,
        2
    )

    return binaryImage

'''影像二值化'''
from cv2 import threshold, THRESH_BINARY_INV, THRESH_BINARY
def to_binary(img, clp=230, mode=THRESH_BINARY_INV):
    
    binary = threshold(img, clp, 255, mode)[1]
    
    return binary

'''影像膨脹化'''
from cv2 import dilate
from numpy import ones, uint8
def to_dilate(image, ker=(3, 3)):
    
    kernel = ones(ker, uint8)
    dilation = dilate(image, kernel)
    
    return dilation

'''影像侵蝕化'''
from cv2 import erode
def to_erode(image, ker=(3, 3)):
    
    kernel = ones(ker, uint8)
    erosion = erode(image, kernel)
    
    return erosion

'''取得僅有線條的影像'''
from cv2 import getStructuringElement, MORPH_RECT, morphologyEx, MORPH_OPEN, MORPH_CLOSE, imwrite
from pathlib import Path as path
def to_lines(img):
    
    rows, cols = img.shape # 圖片長,寬
    
    '''進行垂直開運算，取得直線'''
    kernel = getStructuringElement(MORPH_RECT, (1, rows // 20))
    dilated_row = morphologyEx(img, MORPH_OPEN, kernel) # 影像處理結果(直線)
    
    '''進行水平開運算，取得橫線'''
    kernel = getStructuringElement(MORPH_RECT, (cols // 40, 1))
    dilated_col = morphologyEx(img, MORPH_OPEN, kernel) # 影像處理結果(橫線)
    
    return (dilated_row, dilated_col)

'''取得僅有文字的影像'''
from cv2 import add, subtract
def to_digits(img):
    
    height, width = img.shape[:2]
    
    # 影像二值化
    binary = to_adaptivebinary(img)
    
    # 取得直線影像與橫線影像
    dilated_row, dilated_col = to_lines(binary)
    
    '''直線橫線圖像相加，取得表格輪廓'''
    excel = add(dilated_col, dilated_row)

    '''二值化圖像與表格輪廓相減，移除表格輪廓'''
    digits = subtract(binary, excel)

    '''進行開運算，去除雜訊'''
    kernelSize = (2, 2)
    new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    morphOpen = morphologyEx(digits, MORPH_OPEN, new_kernel)
    
    '''進行閉運算，修復斷裂文字'''
    kernelSize = (2, 2)
    new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    morphClose = morphologyEx(morphOpen, MORPH_CLOSE, new_kernel)
    
    return morphClose

'''取得直線橫線相交點'''
from cv2 import bitwise_and
from numpy import where, rot90
def get_intersect(image):
    
    # 影像二值化
    binary = to_adaptivebinary(image)
    
    # 取得直線影像與橫線影像
    dilated_row, dilated_col = to_lines(binary)
    
    '''直線橫線圖像取AND，以兩張圖片皆為白色的方式取得直線與橫線的交點'''
    bitwiseAnd = bitwise_and(dilated_col, dilated_row)
    dilation = to_dilate(bitwiseAnd, (7, 7))

    coordianate = where(bitwiseAnd > 0)

    return coordianate

def corner_fliter(array):

    # 排序座標數值，將接近值篩選掉取最大座標
    sort_pos = sort(array)

    index = 0
    selected = []
    length = len(sort_pos)-1
    while index<length:
        index+=1

        # 將相距小於8的值跳過
        while sort_pos[index + 1] - sort_pos[index] < 8:
            index+=1
            if index>=length:
                break
        selected.append(sort_pos[index])

    return selected

'''篩選相交點座標'''
from numpy import sort
def get_corners(image):

    height, width = image.shape[:2]
    ys, xs = get_intersect(image)

    coordianate = []

    '''X軸'''
    coordianate.append(corner_fliter(xs))
    

    '''Y軸'''
    
    coordianate.append(corner_fliter(ys))
    
    return coordianate

'''圖像切割'''
from cv2 import cvtColor, COLOR_BGR2GRAY, fastNlMeansDenoising
def split_image(image):
    
    grayImage = cvtColor(image, COLOR_BGR2GRAY)
    
    croppedImages = []
    
    # 圖像降噪
    denoisedImage = fastNlMeansDenoising(grayImage, h=10)
    
    # 取得數字影像
    digitImage = to_digits(denoisedImage)
    
    # 取得裁切點
    x_point_arr, y_point_arr = get_corners(denoisedImage)

    # 裁切
    for i in range(1, len(y_point_arr)):
        temp = []
        for j in range(1, len(x_point_arr)):
            x1, x2, y1, y2 = map(int, (x_point_arr[j-1], x_point_arr[j],\
                                       y_point_arr[i-1], y_point_arr[i]))
            cropImage = digitImage[y1:y2, x1:x2]
            temp.append(cropImage)
        croppedImages.append(temp)

    return croppedImages

'''標記數字位置'''
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect, rectangle
def get_digits(image):
    
    boundboxes = []
    
    kernelSize = (2, 2)
    new_kernel = getStructuringElement(MORPH_RECT, kernelSize)
    final = morphologyEx(image, MORPH_OPEN, new_kernel)

    dilation = to_dilate(final)
    
    (contours, _) = findContours(dilation, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, width, height = boundingRect(cnt)
        boundboxes.append(((x, y), (x+width, y+height)))
        
    return sorted(boundboxes)

'''個別分割文字影像'''
def digit_image(image):
    
    croppedImages = []
    
    boundboxes = get_digits(image)
          
    for b in boundboxes:
        
        x, y, bwidth, bheight = b[0][0], b[0][1], b[1][0], b[1][1]
        
        cropImage = image[y:bheight, x:bwidth]
        
        croppedImages.append(cropImage)
        
    return croppedImages

'''調整影像'''
from cv2 import copyMakeBorder, BORDER_CONSTANT, mean, INTER_NEAREST, COLOR_GRAY2BGR
from numpy import ceil
def to_revision(image):
    
    binary = to_binary(image, mode=THRESH_BINARY_INV)
    
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
        binary,
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
    
'''建立整個待辨識的影像資料'''
def split_digits(image):
    
    cp = split_image(image)
    
    digits = []
    for i in range(len(cp)):
        temp=[]
        for j in range(len(cp[i])):
            digtemp=[]
            for digit in digit_image(cp[i][j]):
                final=to_revision(digit)
                digtemp.append(final)
            temp.append(digtemp)
        digits.append(temp)
    
    return digits

'''辨識機器模型讀取'''
from keras.models import load_model
from numpy import ndarray, float32, asarray

# Load the model
model = load_model('keras_model.h5')


'''產生辨識結果'''
from numpy import ndarray, float32, asarray
def predict(image):
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = ndarray(shape=(1, 224, 224, 3), dtype=float32)

    #turn the image into a numpy array
    image_array = asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    return prediction


'''產生整個辨識完畢的資料'''
from numpy import argmax
def create_prediction_data(digits):
    
    data = []
    
    for row in range(len(digits)):
        rowData = []
        for img in range(len(digits[row])):
            strData = ''
            for d in range(len(digits[row][img])):
                
                prediction = predict(digits[row][img][d])
                
                if prediction.max() > 0.7:
                    prediction = argmax(prediction)
                    strData += str(prediction if prediction < 10 else '.')
                
            rowData.append(strData)
        
        data.append(rowData)
        
    return data

'''轉CSV'''
from pandas import DataFrame
def toCSV(data, fileName):
    data = DataFrame(data)
    data.to_csv(fileName, index = False)
    
'''輸入環節'''
predictData = create_prediction_data(split_digits(imread(input("請輸入影像位置："))))

toCSV(predictData)