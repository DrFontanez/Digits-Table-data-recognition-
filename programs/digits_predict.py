# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:56:39 2022

@author: 6B01
"""
from keras.models import load_model
from numpy import ndarray, float32, asarray, argmax


# Load the model
model = load_model('keras_model.h5')

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

def create_prediction_data(digits):
    
    data = []
    print('建立辨識結果中...')
    
    for row in range(len(digits)):
        rowData = []
        for img in range(len(digits[row])):
            strData = ''
            for d in range(len(digits[row][img])):
                
                prediction = predict(digits[row][img][d])
                
                if prediction.max() > 0.6:
                    prediction = argmax(prediction)
                    strData += str(prediction if prediction < 10 else '.')
                
            rowData.append(strData)
        
        data.append(rowData)
    return data