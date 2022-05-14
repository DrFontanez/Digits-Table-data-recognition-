# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:00:28 2022

@author: 6B01
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from cv2 import imread
from programs.digits_process import split_digits
from programs.digits_predict import create_prediction_data
from pandas import DataFrame

imageName=input()[1:-1]

digits = split_digits(imread(imageName))
data = create_prediction_data(digits)
print('輸出csv檔中...')
DataFrame(data).to_csv("{}.csv".format(imageName), index=False, header=False)

data = open("{}.csv".format(imageName))
data = [line.strip() for line in data]
print(data)
print("完成")

# digits=get_digits(imread(imageName))

# output=""
# for i in digits:
#     prediction=predict(cvtColor(i, COLOR_GRAY2BGR))
#     output+=str(prediction if prediction < 10 else '.')
    
# print(output)