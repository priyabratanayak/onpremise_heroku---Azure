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
import base_model_transfer
import transfer_learning_from_basemodel
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
    st.session_state['tl_model_name']=None
    model_names =[];
    placeholder=st.empty()  
    temp=r'config\customers.txt'
    with open(temp) as f:
        contents = f.readlines()
        for line in contents:
            if "Base".lower() in line.lower():
                if '\n'  in line :
                    model_names.append(line[:line.index("\n")])
                else:
                    model_names.append(line)
                
                
    st.session_state['tl_chkbox_csv_file']=False 
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
        
        # model_names =[];
        # temp=os.listdir('./model')
        # for data in temp:
        #     model_names.append(data)
            
        

         # Defaults to 'text/plain'
        with open('Template\\template.csv', 'rb') as f:
           st.download_button('Download Template', f, file_name='template.csv')  # Defaults to 'application/octet-stream'
        st.markdown("""---""")
        with st.form(key="magic"):
            
            #st.write(os.getcwd())
            if 'tl_counter_magic' not in st.session_state:
                st.session_state['tl_counter_magic']=0
            else:
                st.session_state['tl_counter_magic']=0
            try:
                template_data=pd.read_csv('Template\\template.csv')
            except:
                pass
            col1,col2,col3,col4=st.columns([1.5,0.2,1.5,0.5])
            
            with col1:
                Analyse_CSV = st.empty()
                st.session_state['tl_chkbox_csv_file']=Analyse_CSV.checkbox('Use CSV/Excel File For Transfer Learning',False)
                uploaded_file = st.file_uploader("Upload your Data File in CSV/XLS/XLSX Format",type=['csv','xlsx','xls'])#type=['png','jpeg']
                
                if uploaded_file is not None:
                  user_data =None
                  try:
                      user_data=pd.read_csv(uploaded_file)
                  except:
                      user_data=pd.read_excel(uploaded_file)
                  
                  st.session_state['tl_user_data']=user_data
                
            with col2:
                
                st.write("  OR")
            with col3:
                 
                 st.session_state['tl_data_external'] = st.text_input('GIT url To download csv File.(Click on View raw and Copy the URL)')
                 if len(model_names)>0:
                     st.session_state['tl_model_name'] = st.selectbox(
                    'Select the Desired Model',
                         model_names)
                 else:
                     st.subheader("No Model is Avaliable to Select")    
            with col4:
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
               
                flag=st.form_submit_button('Train')
       
            if flag:
                st.session_state['tl_status_placeholder'].warning("Training Satrted...")
                transfer_learning_from_basemodel.app()
               
                
                            
        if 'tl_status_placeholder' not in st.session_state:
            st.session_state['tl_status_placeholder']=st.empty()
        if 'tl_train_header' not in st.session_state:
              st.session_state['tl_train_header']=st.empty() 
        
        if 'tl_train_result' not in st.session_state:
             st.session_state['tl_train_result']=st.empty() 
        if 'tl_train_plot' not in st.session_state:
             st.session_state['tl_train_plot']=st.empty() 
       
        
                       
    else:
        placeholder.warning("Kindly Login To Access The Page")
       
