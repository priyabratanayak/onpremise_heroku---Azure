import pandas as pd
import datetime as dt
import os
import connect
import copy
import streamlit as st
import numpy as np
import time
import statsmodels.api as sm
import streamlit.components.v1 as components
import base64
import glob
import requests
from bs4 import BeautifulSoup
from AccessValidation import Access
import predict_base_model
import os
m = st.markdown("""
<style>
div.row_heading.level0.row0.css-bhu10j.e11ks2pe4  {
    background-color:none;
    text-align:center;
    textalign:center;
}

div.col_heading.level0.col0.css-bhu10j.e11ks2pe3

{
 background-color:none ;
 color:none;
 }
div.blank.css-iipcjh.e11ks2pe2 {
    background-color: none;
button.css-bt3ecq.edgvbvh5
{
 background-color: #0F0104;
 }
.css-iipcjh
{
 }
.st-fs
{
 background-color:none ;
 }

}
</style>""", unsafe_allow_html=True)
def app():

    
    placeholder=st.empty()   
    if True or st.session_state['userid']!="":
        st.success("Logged in as "+st.session_state.userid)
        
        st.write("""
        # Emperical Findings
        """)
        st.markdown("""---""")
        # os.chdir("C:/Users/10689029/OneDrive - LTI/Documents/AI/ML/hackathon/onpremise_git")
        df = pd.read_excel(io="report/Analysis_accuracy.xlsx")
        df['Train Accuracy ( % )'] = df['Train Accuracy ( % )'].astype("string")
        df['Test Accuracy ( % )'] = df['Test Accuracy ( % )'].astype("string")
        df['# Epoch'] = df['# Epoch'].astype("string")
        df=df.fillna('')
        df=df.rename(index=lambda s: s + 1)
        df_cust1=df.head(1)
        #df_cust1=df_cust1.rename(index=lambda s: s + 1)
        df_cust2=df.iloc[1:3,:]
        #df_cust2=df_cust1.rename(index=lambda s: s + 1)
        df_cust3=df.iloc[3:,:]
        #df_cust3=df_cust1.rename(index=lambda s: s + 1)
        st.subheader("Customer 1")
        st.write(df_cust1)
        st.subheader("Customer 2")
        st.write(df_cust2)
        st.subheader("Customer 3")
        st.write(df_cust3)
    else:
        placeholder.warning("Kindly Login To Access The Page")
   
    
       
