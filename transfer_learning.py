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


def app():
    placeholder=st.empty()   
    
    if True or st.session_state['userid']!="":
        st.success("Logged in as "+st.session_state.userid)
        
        st.write("""
        # Transfer Learning
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
        import os
        model_names =[];
        temp=os.listdir('./Model')
        for data in temp:
            model_names.append(data)
            
        

         # Defaults to 'text/plain'
        with open('Template\\template.csv', 'rb') as f:
           st.download_button('Download Template', f, file_name='template.csv')  # Defaults to 'application/octet-stream'
        st.markdown("""---""")
        with st.form(key="magic"):
            
            #st.write(os.getcwd())
            if 'counter_magic' not in st.session_state:
                st.session_state['counter_magic']=0
            else:
                st.session_state['counter_magic']=0
            try:
                template_data=pd.read_csv('Template\\template.csv')
            except:
                pass
            col1,col2,col3=st.columns([2,1,0.5])
            with col1:
                Analyse_CSV = st.empty()
                csv=Analyse_CSV.checkbox('Analyse CSV File')
                uploaded_file = st.file_uploader("Upload your Data File in CSV Format",type=['csv'])#type=['png','jpeg']
                if uploaded_file is not None:
                  st.write(uploaded_file)
                  #uploaded_file="C:\\Users\\PRIYABRATANAYAK\\Documents\\Python Tutorial\\sharereport\\Front End\\Data\\My New File.csv"
                  user_data = pd.read_csv(uploaded_file)
                
            with col2:
                Analyse_Oddlogic = st.empty()
                Oddlogic=Analyse_Oddlogic.checkbox('Analyse Oddlogic Account')
                if len(model_names)>0:
                    option = st.selectbox(
                   'Select the Desired Model',
                        model_names)
                else:
                    st.subheader("No Model is Avaliable to Select")
                
            #with col3:
               # Analyse_both = st.empty()
                #both=Analyse_both.checkbox('Analyse Both CSV and Oddlogic Account')
            with col3:
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                flag=st.form_submit_button('Train')
       
            if flag:
               
                   
                    
                
                data="""\
                    Function to Calculate Slope
                    **Note:**
                    
                    """
                st.markdown(data)
                            
                            
                            
                           
                        
    else:
        placeholder.warning("Kindly Login To Access The Page")
       
