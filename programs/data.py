# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:59:51 2022

@author: 6B01
"""
from pandas import DataFrame

def toCSV(data, fileName):
    DataFrame(data).to_csv(fileName, index = False)