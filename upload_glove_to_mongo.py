# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:09:57 2022

@author: biswa
"""

import streamlit as st
import xlsxwriter
from io import BytesIO
from userValidation import SigninDetails
import pandas as pd
import os
import numpy as np
Collection_Credentials="glove"
#oddlogic_Prediction=SigninDetails("mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority","oddlogic")
oddlogic_Prediction=SigninDetails('mongodb+srv://oddlogic:oddlogic@cluster0.h52iyb6.mongodb.net/?retryWrites=true&w=majority',"oddlogic")

oddlogic_Prediction.create_Collection(Collection_Credentials)            

glovevectorfile="glove/glove_150.txt"
name="glove_150.txt"

file_data=open(glovevectorfile,'rb')
data=file_data.read()
print(len(data))
oddlogic_Prediction.upload_gloveFile(data,'glove_150')


