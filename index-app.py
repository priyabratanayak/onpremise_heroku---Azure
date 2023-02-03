import streamlit as st


    
import pandas as pd
import numpy as np
import pickle
import time
import base64
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components
import plotly.graph_objects as go
from  multipage import MultiPage
import login,signup,settings,prediction,training,transfer,report,prediction_tnsr,superimposed,azuretest
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




st.session_state['app'] = MultiPage()    
# Title of the main page

st.session_state['app'].add_page("Login", login.app)
st.session_state['app'].add_page("Signup", signup.app)
st.session_state['app'].add_page("azure", azuretest.app)
st.session_state['app'].add_page("Training", training.app)
st.session_state['app'].add_page("Transfer Learning",transfer.app)
st.session_state['app'].add_page("Superimposed Learning",superimposed.app)
st.session_state['app'].add_page("Prediction",prediction.app)
st.session_state['app'].add_page("Prediction In Cloud",prediction_tnsr.app)
st.session_state['app'].add_page("Report",report.app)
st.session_state['app'].add_page("Settings", settings.app)
st.session_state['app'].run()

