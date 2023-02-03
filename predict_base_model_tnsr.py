
from io import BytesIO
import pickle
import requests
import numpy as np # For mathematical calculations
import pandas as pd # For Dtaa frames
from datetime import datetime
from sklearn.metrics import f1_score,recall_score,precision_score,classification_report,accuracy_score
import sys
import re
from datetime import datetime


import requests
import json
import os
import streamlit as st

from PIL import Image

# Parameters
filename = "base_V11_Test"
MAX_SEQUENCE_LENGTH = 40
# Model Parameters - configurable
path = './files_prediction'
inputpath = './input'
sl_name = 'base' 
ver = 'V11'
summary_col = 'Short description'
final_stop_words=None
labels_index=None

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
    url=""
    from nltk.corpus import stopwords    
    global final_stop_words
    # Model Parameters - Non configurable
    global labels_index
    
    tokenizer=None
    
    with open('./files_prediction/base_V11_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Load Model Components for model '+sl_name)
    
    #-----------------------------------------------------------------------------
    
    if st.session_state['customer'] =="Customer 1":
        url="http://localhost:8601/v1/models/oddlogic/labels/cust1:predict"
        
        labels_index = load_obj('files_prediction/base_V11_labels_dict.pickle')
        
        if st.session_state['chkbox_csv_file']:
            df=st.session_state['user_data']
        else:
            df = pd.read_excel(io=inputpath+"/dataset_base_predict.xlsx")
        if st.session_state['data_external']:
            try:
                df=pd.read_csv( st.session_state['data_external'])
            except:
                df = pd.read_excel(io=inputpath+"/dataset_base_predict.xlsx")
    
    if st.session_state['customer'] =="Customer 2":
        
        if st.session_state['option'] =="Transfered Model":
            url="http://localhost:8602/v1/models/oddlogic/labels/cust2trnsfer:predict"
            
            labels_index = load_obj('./files_prediction/cust2_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=inputpath+"/dataset_cust2_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=inputpath+"/dataset_cust2_predict.xlsx")
    
        if st.session_state['option'] =="Superimposed Model":
            url="http://localhost:8603/v1/models/oddlogic/labels/cust2si:predict"
            labels_index = load_obj('./files_prediction/cust2_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=inputpath+"/dataset_cust2_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=inputpath+"/dataset_cust2_predict.xlsx")
                    
                    
              
    if st.session_state['customer'] =="Customer 3":
        if st.session_state['option'] =="Transfered Model":
            labels_index = load_obj('./files_prediction/customer3_transfered_V11_labels_dict.pickle')
           
            if st.session_state['chkbox_csv_file']:
                df=st.session_state['user_data']
            else:
                df = pd.read_excel(io=inputpath+"/dataset_cust3_predict.xlsx")
            if st.session_state['data_external']:
                try:
                    df=pd.read_csv( st.session_state['data_external'])
                except:
                    df = pd.read_excel(io=inputpath+"/dataset_cust3_predict.xlsx")
    #-----------------------------------------------------------------------------
    starttime = datetime.now()
  
    
    #Load back Tokenizer
    
    #with open('./savedmodels/'+sl_name+'_'+ver+'_tokenizer.pickle', 'rb') as handle:
    #   tokenizer = pickle.load(handle)
        
    
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Tokenizer Loaded Successfully.')
    
    
        
    #labels_index = load_obj('./savedmodels/'+sl_name+'_'+ver+'_labels_dict.pickle')
    
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Label_Index Loaded Successfully.\n')
    
    # if st.session_state['chkbox_csv_file']:
    #     df=st.session_state['user_data']
    # else:
        
    #     df = pd.read_excel(io=inputpath+"/"+filename+".xlsx") #,sheet_name = 'Inc')
    # if st.session_state['data_external']:
    #     try:
    #         user_data=pd.read_csv( st.session_state['data_external'])
    #     except:
    #         df = pd.read_excel(io=inputpath+"/"+filename+".xlsx") #,sheet_name = 'Inc')
        
    
    df.dropna(subset=[summary_col], inplace=True)
    #df.dropna(subset=['SubCategory'], inplace=True)
    print(str(datetime.now())[:19]+'-> '+'Data file shape after removing any missing summary records:',df.shape)
    
    
    stopwordlist = stopwords.words('english')
    stopwordlist.append('would')
    not_stopwords = {'not','up','down','on','off','above','below','between'}
    final_stop_words = [word for word in stopwordlist if word not in not_stopwords]
    
    
    
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Start Data Preprocessing..')
    
    
    
    # Call preprocessing functions
    df[summary_col+'_original'] =  df[summary_col]
    df[summary_col]=list(map(replaceNumber,df[summary_col]))
    df[summary_col]=list(map(replaceINC,df[summary_col]))
    df[summary_col]=list(map(replaceREQ,df[summary_col]))
    df[summary_col]=list(map(replaceURL,df[summary_col]))
    df[summary_col]=list(map(replaceEmail,df[summary_col]))
    df[summary_col]=list(map(replacePO,df[summary_col]))
    
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Data Preprocessing complete..')
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Start Prediction..')
    
    
    #### Predict Target for Unseen Data
    sample_rec=df[summary_col] 
    seq = tokenizer.texts_to_sequences(sample_rec)
    
    sample_rec_seq=[]
    for data in seq:
     sample_rec_seq.append( np.pad(seq[0],(MAX_SEQUENCE_LENGTH,0),mode='constant', constant_values=0))
    sample_rec_seq=np.array(sample_rec_seq)
    
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Get class probabilities for each input..')
    
    #url="http://localhost:8605/v1/models/oddlogic:predict"
    
    convertd_list=[]
    for data in sample_rec_seq:
        convertd_list.append(data.tolist())
    if len(convertd_list)==1:
        convertd_list=list(convertd_list)
    payload={
        "instances":convertd_list
    }
    response=requests.post(url,json=payload)
    
    
    pred =response.text# loaded_model.predict(sample_rec_seq[0])
    pred_json=json.loads(response.text)
    
    res_prob=list(map(max,pred_json['predictions']))
    
    print(str(datetime.now())[:19]+'-> '+'Get class label index for each input..')
    res=list(map(np.argmax,pred_json['predictions']))
    
    labels_pred=list(map(get_label,res))
    
    
    # In[29]:
    
    
    df['Category_Predicted'] = labels_pred
    #df['CatSubcat_CF'] = res_prob
    #df['CF'] = list(map(lambda x:round(x*100,2),df['CatSubcat_CF']))
    df['Confidence_Probability'] = list(map(lambda x:round(x*100,2),res_prob))
    
    df['ProcessType'] = 'ML'
    df['keyword_used'] = ''
    
    #df = df.append(key_mapped,ignore_index = True, sort=True)[key_mapped.columns.tolist()]
    
    #print(str(datetime.now())[:19]+'-> '+'Keywords percent      ::',round((df[df['Confidence_Probability'] == 100].shape[0]/df.shape[0])*100,2))
    print(str(datetime.now())[:19]+'-> '+'Overall >90 CL percent::',round((df[df['Confidence_Probability'] >= 90].shape[0]/df.shape[0])*100,2))
    print(str(datetime.now())[:19]+'-> '+'Overall >80 CL percent::',round((df[df['Confidence_Probability'] >= 80].shape[0]/df.shape[0])*100,2))
    
    
    # In[ ]:
    
    
    print(str(datetime.now())[:19]+'-> '+'Save predictions to file.')
    outfile = filename+'_ML_Prediction_'+ver+'.xlsx'
    #df[column_order].to_excel('.\\..\\OutputData\\'+outfile, index=False,encoding ='utf8')
    df.to_excel(path+'/'+outfile, index=False,encoding ='utf8')
    
    
    # In[ ]:
    
    
    print('\n***********************************\n')
    print(str(datetime.now())[:19]+'-> '+'Predictions saved to the file               :',outfile)
    endtime = datetime.now()
    tm = str(endtime - starttime)
    st.session_state['tnsnr_status_predict_placeholder'].warning(str(datetime.now())[:19]+'-> '+'Total time taken for prediction(Hr:Min:Sec) :'+tm[:7])
    print(str(datetime.now())[:19]+'-> '+'Total number of rows processed              :',df.shape[0])
    print('\n***********************************\n')
    
    if "Unnamed: 0" in df:
        df.drop(['Unnamed: 0'], axis=1)
    if "Unnamed: 1" in df:
        df.drop(['Unnamed: 1'], axis=1)
    
    st.session_state['tnsnr_status_predict_placeholder'].warning("Prediction Finished.")
    my_expander = st.session_state['tnsnr_predict_result'].expander(label='Prediction Summary')
    with my_expander:    
     st.write(str(datetime.now())[:19]+'-> '+'Predictions saved to the file               :\n'+outfile)
     st.write(str(datetime.now())[:19]+'-> '+'Total number of rows processed              :',df.shape[0])
    st.session_state['tnsnr_predict_header'].subheader('Prediction Result')
    st.session_state['tnsnr_predict_df'].write(df)
    st.session_state['tnsnr_user_data']=None
    
   
    
