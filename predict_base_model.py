
import pickle

#import tf.keras

import numpy as np # For mathematical calculations
import pandas as pd # For Dtaa frames

from datetime import datetime
from sklearn.metrics import f1_score,recall_score,precision_score,classification_report,accuracy_score
import sys
import re
from datetime import datetime
import streamlit as st

    

final_stop_words=None
import os

# Model Parameters - configurable
filename = "Final_Prediction"
path = './input'
sl_name = 'base' 
ver = 'V11'
summary_col = 'Short description'
labels_index=None

MAX_SEQUENCE_LENGTH = 40




# In[ ]:


# Load Label_Index
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
# Define preprocessing functions
def replaceNumber(x):
    return re.sub('[^<a-zA-Z0-9>][\d]+',' #Nembor#',str(x))

def replaceINC(x):
    return re.sub("INC+\d+","#IncidentNum#",str(x))

def replaceREQ(x):
    return re.sub("REQ+\d+","#RequestNum#",str(x))

def replaceURL(x):
    return (re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',"#URL#",str(x)))

def replaceEmail(x):
    return (re.sub("[\w\.-]+@[\w\.-]+\.\w+","#EmailID#",str(x)))

def replacePO(x):
    return (re.sub("PO+\d+","#PurchaseOrder#",str(x)))

def replacestopword(x):
    sline = [word.lower() for word in str(x).split() if word.lower() not in final_stop_words]
    strline = ' '.join(sline)
    return strline

def replaceothers(x):
    line=re.sub(r'\[MP\]\s*\[DOM\].*\[AN\]'," ",str(x))# removes [MP]  [DOM] anything in between [AN]. [MP]  [DOM] INTERNAL [AN] 
    return (re.sub(r' (\[AD\])|(\[ADN\])|(\[AN\])' ," ",line))# removes [AN],[ADN],[AN]
# Function to return Label from the index
def get_label(x):
    return list(labels_index.keys())[list(labels_index.values()).index(x)]
def app():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences # For padding the text sequences
    import nltk
    from nltk.corpus import stopwords
    global final_stop_words
    loaded_model=None
    df=None
    global labels_index    
    tokenizer=None
    st.session_state['status_predict_placeholder'].warning("Predicting Data...........")
    with open('./files_prediction/base_V11_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    if st.session_state['customer'] =="Customer 1":
        
        loaded_model = load_model(r'files_prediction\best_model.hdf5')
        
        labels_index = load_obj(r'files_prediction\base_V11_labels_dict.pickle')
        
        if st.session_state['chkbox_csv_file']:
            df=st.session_state['user_data']
        else:
            df = pd.read_excel(io=path+"/dataset_base_predict.xlsx")
        if st.session_state['data_external']:
            try:
                df=pd.read_csv( st.session_state['data_external'])
            except:
                df = pd.read_excel(io=path+"/dataset_base_predict.xlsx")
    
    if st.session_state['customer'] =="Customer 2":
        if st.session_state['option'] =="Transfered Model":
            loaded_model = load_model('./files_prediction/'+'cust2_transferred_model.hdf5')
            labels_index = load_obj('./files_prediction/cust2_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=path+"/dataset_cust2_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=path+"/dataset_cust2_predict.xlsx")
    
        if st.session_state['option'] =="Superimposed Model":
            loaded_model = load_model('./files_prediction/'+'cust2_superimposed_model.hdf5')
            labels_index = load_obj('./files_prediction/cust2_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=path+"/dataset_cust2_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=path+"/dataset_cust2_predict.xlsx")
                    
                    
              
    if st.session_state['customer'] =="Customer 3":
        if st.session_state['option'] =="Transfered Model":
            loaded_model = load_model('./files_prediction/'+'customer3_transfered_model.hdf5')
            labels_index = load_obj('./files_prediction/customer3_transfered_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=path+"/dataset_cust3_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=path+"/dataset_cust3_predict.xlsx")
    
        if st.session_state['option'] =="Superimposed Model":
            loaded_model = load_model('./files_prediction/'+'customer3_superimposed_new_model.hdf5')
            labels_index = load_obj('./files_prediction/customer3_transfered_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=path+"/dataset_cust3_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=path+"/dataset_cust3_predict.xlsx")
    

   
    # Drop the rows with empty cells in summary column
    df.dropna(subset=[summary_col], inplace=True)
    
    # In[ ]:
    
    
    #Exclude stopwords as per ur need
    stopwordlist = stopwords.words('english')
    stopwordlist.append('would')
    not_stopwords = {'not','up','down','on','off','above','below','between'}
    final_stop_words = [word for word in stopwordlist if word not in not_stopwords]
    
    
    # In[ ]:
    
    
    
    print(str(datetime.now())[:19]+'-> '+'Start Data Preprocessing..')
    
    
    # In[ ]:
    
    
    # Call preprocessing functions
    df[summary_col+'_original'] =  df[summary_col]
    df[summary_col]=list(map(replaceNumber,df[summary_col]))
    df[summary_col]=list(map(replaceINC,df[summary_col]))
    df[summary_col]=list(map(replaceREQ,df[summary_col]))
    df[summary_col]=list(map(replaceURL,df[summary_col]))
    df[summary_col]=list(map(replaceEmail,df[summary_col]))
    df[summary_col]=list(map(replacePO,df[summary_col]))
    
    print(str(datetime.now())[:19]+'-> '+'Data Preprocessing complete..')
    print(str(datetime.now())[:19]+'-> '+'Start Prediction..')
    
    
    # In[ ]:
    
    
    #### Predict Target for Unseen Data
    sample_rec=df[summary_col] 
    seq = tokenizer.texts_to_sequences(sample_rec)
    sample_rec_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  
    pred = loaded_model.predict(sample_rec_seq)
    res_prob=list(map(max,pred.tolist()))
   
    res=list(map(np.argmax,pred.tolist()))
   
    labels_pred=list(map(get_label,res))
    
    
    df['Predicted_Assignment_Group'] = labels_pred
    
    df_op = df.filter(['Number', 'Short description','Predicted_Assignment_Group'])
    df_op.rename(columns = {'Number':'Ticket Number'}, inplace = True)
   
    st.session_state['status_predict_placeholder'].warning("Prediction Successful.")
    st.session_state['predict_result'].write("Summary")
    st.session_state['predict_header'].write(df_op)
