# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:37:27 2022

@author: biswa
"""

import streamlit as st
import xlsxwriter
from io import BytesIO
from userValidation import SigninDetails
import pandas as pd
import os
import numpy as np
Collection_Credentials="input"
oddlogic_Prediction=SigninDetails("mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority","oddlogic")

oddlogic_Prediction.create_Collection(Collection_Credentials) 

tktdata = pd.read_excel(r'Input\hon.xlsx',sheet_name='Sheet1')
tktdata=tktdata.head(1000)


tktdata = tktdata [['Number','Short description','Assignment group','Issue Tag']]

traincolumn="Short description"
predictedcolumn="Assignment group"
# Drop the rows with empty cells in summary column
tktdata.dropna(subset=[traincolumn], inplace=True)
tktdata.dropna(subset=[predictedcolumn], inplace=True)  
unique_labs = np.unique(tktdata[predictedcolumn])
labels_index={}  # dictionary mapping label name to numeric id

for lab in unique_labs:
    label_id = len(labels_index)
    labels_index[lab] = label_id

tktdata['Codes'] = list(map(lambda x: labels_index[x], tktdata[predictedcolumn]))

tktdata_converted=pd.DataFrame.to_dict(tktdata,orient='records')
print(type(tktdata_converted))
print(unique_labs.shape)
oddlogic_Prediction.clear_inputdata()
oddlogic_Prediction.upload_inputdata(tktdata_converted)
