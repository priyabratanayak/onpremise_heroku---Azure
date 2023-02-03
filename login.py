
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
#from multipage import MultiPage
#from selenium import webdriver
import pymongo
import os
import datetime
import time
from PIL import  Image
import bcrypt
import pytz
from pathlib import Path

#os.chdir(r'C:\Users\10689029\OneDrive - LTI\Documents\AI\ML\hackathon\onpremise_git')
from userValidation import SigninDetails
import sys


#import pymongo

#conn = sqlite3.connect('data.db')
#c = conn.cursor()

class Access():
    def __init__(self,DEFAULT_CONNECTION_URL = "mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority",DB_NAME=None):
        self.DEFAULT_CONNECTION_URL = DEFAULT_CONNECTION_URL
        self.DB_NAME = DB_NAME
        self.collection_dict={}
        # Establish a connection with mongoDB
        self.client = pymongo.MongoClient(self.DEFAULT_CONNECTION_URL)
        # Create a DB
        dbnames = self.client.list_database_names()
        #if self.DB_NAME not in dbnames:
        self.dataBase = self.client[self.DB_NAME]
        self.cl_price_df = pd.DataFrame()
        self.ohlcv = {}
        self.ohlcv_df = pd.DataFrame()
        self.ohlcv_day = {}
    def create_Collection(self,*argv):
        collection_list = self.dataBase.list_collection_names()
        for name in argv:
                 #if name not in collection_list:
                 self.collection_dict[name]=self.dataBase[name]
    def returncollection(self,collectionname):
        return self.collection_dict[collectionname]
    
    def add_access(self,toAcess,fromAccess,*argv):
        message=[]
        for name in argv:
            if name=="Access_Given":
                accessgivenlist=[]
                if len(list(self.collection_dict["Access_Given"].find({"owner":toAcess})))>0:
                     for x in (self.collection_dict["Access_Given"].find({"owner":toAcess})):
                         accessgivenlist.extend(x['list'])
                     if toAcess not in accessgivenlist:
                         accessgivenlist.append(fromAccess)
                     self.collection_dict["Access_Granted"].update({"owner":toAcess},{"$set":{"list":accessgivenlist}})
                
                else:
                    accessgivenlist.append(fromAccess)
                    self.collection_dict["Access_Given"].insert({"owner":toAcess,"list":accessgivenlist})
                
                message.append("Access Provided Successfully")
            
            if name=="Access_Granted":
                accesslist=[]
                if len(list(self.collection_dict["Access_Granted"].find({"owner":fromAccess})))>0:
                     for x in (self.collection_dict["Access_Granted"].find({"owner":fromAccess})):
                         accesslist.extend(x['list'])
                                              
                     if toAcess not in accesslist:
                        accesslist.append(toAcess)
                     self.collection_dict["Access_Granted"].update({"owner":fromAccess},{"$set":{"list":accesslist}})
                
                else:
                    accesslist.append(toAcess)
                    self.collection_dict["Access_Granted"].insert({"owner":fromAccess,"list":accesslist})
                
               
           
        return message
    def revoke_access(self,toAcess,fromAccess,*argv):
        message=[]
        for name in argv:
            if name=="Access_Given":
                accessgivenlist=[]
                if len(list(self.collection_dict["Access_Given"].find({"owner":toAcess})))>0:
                     for x in (self.collection_dict["Access_Given"].find({"owner":toAcess})):
                         accessgivenlist.extend(x['list'])
                     if len(accessgivenlist)>0:
                         
                         accessgivenlist.remove(fromAccess)
                         self.collection_dict["Access_Granted"].update({"owner":toAcess},{"$set":{"list":accessgivenlist}})
                    
                
                
            if name=="Access_Granted":
                accesslist=[]
                if len(list(self.collection_dict["Access_Granted"].find({"owner":fromAccess})))>0:
                     for x in (self.collection_dict["Access_Granted"].find({"owner":fromAccess})):
                         accesslist.extend(x['list'])
                     if len(accesslist)>0:                         
                         
                         accesslist.remove(toAcess)
                         self.collection_dict["Access_Granted"].update({"owner":fromAccess},{"$set":{"list":accesslist}})
                    
                
                message.append("Access Revoked Successfully")
            
        return message
    def Delete_UserID(self,username):
        keys=self.collection_dict.keys()
        
        
        for key in keys:
            
            if key =="Credentials":
                if self.collection_dict[key].count_documents({"userid":username})>0:
                        self.collection_dict[key].delete_one({"userid":username})
                        
                        return("User Deleted Successfully")
                else:
                        return ("User Not Found")
   
    def fetch_record(self,collection_Name,key):
        return self.collection_dict[collection_Name].find({"owner":key})
                            
    
    def clear_All_Collections(self):
        keys=self.collection_dict.keys()
        for key in keys:
                self.collection_dict[key].remove({})
    def get_Records_Collection(self,fetch_date,collection_Name,stockName=None):
        return self.collection_dict[collection_Name].find({"$and":[{"Stock":stockName},{"Date_Str":fetch_date}]})
    
        
    
    
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
 



        
        
            

def app():
       
        if 'msg_placeholder' not in st.session_state:
            st.session_state['msg_placeholder']= None
        
        #Collection_Credentials="Credentials"
        #Prediction=SigninDetails("mongodb://localhost:27017/","Model")
        #Prediction.create_Collection(Collection_Credentials)
        st.subheader("Login Section")
        
        with st.form(key="loginform"):
            if len(st.session_state.userid)>0:
                user_placeholder=st.empty() 
                
                new_user = user_placeholder.text_input("User Name",st.session_state.userid)
            else:
                new_user = st.text_input("User Name",value="Oddlogic")
            pw_placeholder=st.empty() 
            new_password = pw_placeholder.text_input("Password",type='password',value="Oddlogic")
            
            col1,col2=st.columns([1,1])
            with col1:
                signin=st.form_submit_button("Sign In")
            with col2:
                signout=st.form_submit_button("Sign Out")
            st.session_state['msg_placeholder']=st.empty()
            if signout:
                st.session_state['msg_placeholder'].success("Logged out successfully...")
                st.session_state['userid']=""
            if signin:
                if len(new_user)==0:
                    st.session_state['msg_placeholder'].warning("Enter User Name")
                elif len(new_password)==0:
                    st.session_state['msg_placeholder'].warning("Enter User Password")
                else:
                    
                    new_user_utf=str(new_user).strip().upper()
                    new_password_utf=str(new_password).strip().encode('utf-8') 
                    
                    Collection_Credentials="Credentials"


                    oddlogic_Prediction=SigninDetails("mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority","oddlogic")
                    
                    oddlogic_Prediction.create_Collection(Collection_Credentials)            
                       
                    result=oddlogic_Prediction.validate_credentials(new_user_utf,new_password_utf) 
                    
                    
                    if result:
                        
                        st.session_state['userid']=new_user
                        
                        st.session_state['msg_placeholder'].success("Logged In as {}".format(st.session_state['userid']))
                        
                        st.balloons()
                    else:
                        
                        st.session_state['msg_placeholder'].warning("Incorrect Username/Password") 
            