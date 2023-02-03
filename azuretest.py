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
import base_model_training
from PIL import Image
def app():
    
    m = st.markdown("""
<style>
div.row_heading.level0.row0.css-bhu10j.e11ks2pe4  {
    background-color:none;
    
}
div.col_heading.level0.col0.css-bhu10j.e11ks2pe3
{
 background-color:none ;
 color:none;
 }
div.blank.css-iipcjh.e11ks2pe2 {
    background-color: #0F0104;
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
    from azureml.core import Workspace, Dataset, Experiment
    
    placeholder=st.empty()   
    st.session_state['chkbox_csv_file']=False 
    if True or st.session_state['userid']!="":
        st.success("Logged in as "+st.session_state.userid)
        
        st.write("""
        # Training
        """)
        st.write(":heavy_minus_sign:" * 47) 
        
        st.write("""
        # 
        This section trains the model based on the **File** uploaded
        Git Link where Models are stored [Model library](https://github.com/SanjuktaBiswal/oddlogic) .
        """)
        st.markdown("""---""")
        
        
        background_color='#F5F5F5'
        
        text = """\
    You can download the Template and fill the Data in the below Format.
    **Note:**
     - Do not change the Column Header(1st Row)
     - **Name** and **Exchange** columns are optional.
     
    
    """ 
        
    
        # -----------------------------------------------------
        # Access the Workspace and Datasets
        # -----------------------------------------------------
        print('Accessing the workspace....')
        #st.write(os.getcwd())
        #files = [f for f in os.listdir('.') if os.path.isfile(f)]
        #for f in files:
        #    st.write(f)
        try:

            ws = Workspace.from_config("./config")
            
            print('Accessing the dataset....')
            az_dataset        = Dataset.get_by_name(ws, 'AdultIncome')    
            df = az_dataset.to_pandas_dataframe()
            df=df.dropna()
            st.write(df)
        except Exception as e:
            st.write(e)
                       
    else:
        placeholder.warning("Kindly Login To Access The Page")
       
