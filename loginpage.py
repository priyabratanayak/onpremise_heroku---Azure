import streamlit as st
import pandas as pd
from pathlib import Path

import os
 
import plotly.express as px
import plotly.graph_objects as go
# Security
#passlib,hashlib,bcrypt,scrypt

import hashlib
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import base64
import time

from multipage import MultiPage
import sys



import login,signup#,settings,Home,report,Home_SeatAllocation,resource_report
from PIL import  Image

try:
    st.set_page_config(layout="wide")
except:
    pass

header=st.container()

timestr=time.strftime('%Y%m%d%H%M%S')
features=st.container()
df_result=None

background_color='#F5F5F5'
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
def text_downloader(raw_text,file,filename):
    csvfile=file.to_csv(index = False)
    b64=base64.b64encode(csvfile.encode()).decode()
    new_filename=filename+"_{}.csv".format(timestr)
    href=f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download File</a>'
    st.markdown(href,unsafe_allow_html=True)



def main():
    """Simple Login App"""
    # Create an instance of the app 
    
    
    if 'msg' not in st.session_state:
            st.session_state['msg']=None
    if 'placeholder_msg' not in st.session_state:
            st.session_state['placeholder_msg']=None
    if 'userid' not in st.session_state:
            st.session_state['userid']=""
    if 'pw' not in st.session_state:
            st.session_state['pw']=""
    if 'app' not in st.session_state:
            st.session_state['app']=""
    if 'app_analysis' not in st.session_state:
            st.session_state['app_analysis']=""
        
    #...........login Page
    if 'clientaccount_access' not in st.session_state:
        st.session_state['clientaccount_access']=[]
    
   
    
    #generate trading session
    
    
    st.session_state['app'] = MultiPage()    
    # Title of the main page
    
    st.session_state['app'].add_page("Login", login.app)
    st.session_state['app'].add_page("Signup", signup.app)
    st.session_state['app'].add_page("Allocation", Home_SeatAllocation.app)
    st.session_state['app'].add_page("Reports", report.app)
    st.session_state['app'].add_page("Resource Reports", resource_report.app)
    st.session_state['app'].add_page("Settings", settings.app)
    st.session_state['app'].run()
    


if __name__ == '__main__':
    #..................Hide "Made with Streamlit" and Hamburger Icon..............
    
    hide_menu_style="""
    <style>
    #MainMenu {visibility:hidden;}
    footer{visibility:hidden;}
    </style>
    """
    st.markdown(hide_menu_style,unsafe_allow_html=True)    
    main()
