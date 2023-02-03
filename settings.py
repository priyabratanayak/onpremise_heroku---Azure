
import plotly.express as px
import plotly.graph_objects as go
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import base64
import time
from PIL import  Image
import os
from pathlib import Path
import datetime as dt
import bcrypt
import pymongo
from AccessValidation import Access
def app():
    if 'info' not in st.session_state:
        st.session_state['info']=None
    placeholder=st.empty()  
    admin_flag=False
    msg=""
    if  st.session_state['userid']!="":
            #st.session_state.userid="priyabrata"
            st.success("Logged in as "+st.session_state.userid)
            if 'access_placeholder' not in st.session_state:
                st.session_state['access_placeholder']= None
            if 'msg_placeholder' not in st.session_state:
                st.session_state['msg_placeholder']= None
            
            Collection_Access_Given="Access_Given"
            Collection_Access_Granted="Access_Granted"
            Collection_Access_Admin="Admin"
            Prediction=Access("mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority","oddlogic")
            Prediction.create_Collection(Collection_Access_Given,Collection_Access_Granted,Collection_Access_Admin)
            if Prediction.returncollection(Collection_Access_Admin).find_one({})["admin"].upper()==st.session_state.userid.upper():#{:st.session_state.userid.upper()})
                admin_flag=True    
            
            with st.form(key="access"):
                
                #st.session_state.userid="priyabrata".upper()
                
                
                access_col1,access_col2,access_col3,access_col4=st.columns([1,1,1,1])   
                with access_col1:
                    user_placeholder=st.empty() 
                    new_user = user_placeholder.text_input("Access")
                with access_col2:
                    
                    list_owner=[]
                    cur=Prediction.returncollection(Collection_Access_Given).find({"owner":st.session_state.userid.upper()})
                    for x in range(0,len(list(Prediction.returncollection(Collection_Access_Given).find({"owner":st.session_state.userid.upper()})))):
                            
                            list_owner.extend(cur[x]['list'])
                      
                    option = st.selectbox(
                    'Access Given by Admin',
                    tuple(list_owner))
                    
                
                    
                with access_col3:
                    pass
                    
                    list_client=[]
                    cur=Prediction.returncollection(Collection_Access_Granted).find({"owner":st.session_state.userid.upper()})
                    for x in range(0,len(list(Prediction.returncollection(Collection_Access_Granted).find({"owner":st.session_state.userid.upper()})))):
                            
                            list_client.extend(cur[x]['list'])
                    list_client=list(set(list_client))
                    
                    try:
                        list_client.remove(st.session_state.userid.upper()) 
                    except Exception as e:
                        print(e)
                    option = st.selectbox(
                    'List of Account Access Given By You',
                    tuple(list_client))
                    
                with access_col4:
                    
                    st.write("")
                    st.write("")
                    refresh=st.form_submit_button("Refresh")
                col1,col2=st.columns([1,1])
                
                with col1:
                    Provide=st.form_submit_button("Provide Access")
                with col2:
                    Revoke=st.form_submit_button("Revoke Access")
                st.session_state['msg_placeholder']=st.empty()
                if refresh:
                    pass
                if Revoke:
                    if not admin_flag:
                              st.session_state['msg_placeholder'].error("You do not Have Admin Access")
                    else:   
                        
                        msgs=Prediction.revoke_access(str(option).upper(),st.session_state.userid.upper(),Collection_Access_Given,Collection_Access_Granted)
                              
                        st.session_state['msg_placeholder'].success(' '.join(msgs))
                   
                    
                if Provide:
                          if not admin_flag:
                              st.session_state['msg_placeholder'].error("You do not Have Admin Access")
                          elif st.session_state.userid.upper()!=new_user.upper():
                              if len(str(new_user).upper())==0:
                                 st.session_state['msg_placeholder'].warning("Access field is blank")
                              else:
                                  msgs=Prediction.add_access(str(new_user).upper(),st.session_state.userid.upper(),Collection_Access_Given,Collection_Access_Granted)
                                  st.session_state['msg_placeholder'].success(' '.join(msgs))
                st.session_state['msg_placeholder']=st.empty()
            
               
                
    else:
        placeholder.warning("Kindly Login To Access The Page")