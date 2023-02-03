from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re,os
from nltk.corpus import stopwords
final_stop_words=None
labels_index=None
from PIL import Image
import matplotlib.pyplot as plt
# In[2]:
def save_obj(obj,name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# Define preprocessing functions

def replaceNumber(x):
    return re.sub('[^<a-zA-Z0-9>][\d]+',' #Nembor#',str(x))

def replacemultipleNembor(x):
    return re.sub('(#Nembor#){2,}',' #Nembor#',str(x))

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
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    from tensorflow.keras.layers import Input,Dense,Embedding,LSTM
    import tensorflow_addons as tfa
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
    from tensorflow.keras.models import Model
   # os.chdir(r"C:\Users\10689029\OneDrive - LTI\Documents\AI\ML\hackathon\onpremise_git")
    global final_stop_words
    global labels_index
    starttime = datetime.now()
    print('Starting time:',starttime)
    sl_name = 'superimposed' 
    ver = 'V11' ##for model file name--sanjukta
    traincolumn="Short description"
    predictedcolumn="Assignment group"
    #glovevectorfile='./glove/glove_150.txt'
    st.session_state['si_status_placeholder'].warning("Fetching Customer Data...")
    
    tktdata =None# pd.read_excel(r'input\JCI.xlsx')
    if st.session_state['si_chkbox_csv_file']:
        tktdata=st.session_state['si_user_data']
    else:
        tktdata=pd.read_excel('./input/Banner_customer2.xlsx') #sanjukta
        
    if st.session_state['si_data_external']:
        try:
            tktdata=pd.read_csv( st.session_state['si_data_external'])
        except:
            tktdata=pd.read_excel('./input/Banner_customer2.xlsx') 
            
    
    st.session_state['si_status_placeholder'].warning("Finished Fetching Customer Data...")
    #tktdata = tktdata.head(3000)
    tktdata = tktdata.head(3000)
    
    
    
    # In[5]:
    
    
    # Change the column names to standard column name
    tktdata.columns = ['Number','Short description','Assignment group','Issue Tag']
    print (tktdata.head())
    
    # Drop the rows with empty cells in summary column
    tktdata.dropna(subset=[traincolumn], inplace=True)
    tktdata.dropna(subset=[predictedcolumn], inplace=True)
    
    print('Data file shape after removing any missing tkt_notes records and Assigned Group:',tktdata.shape)
    
    
    # In[6]:
    
    
    # Construct Lable Index dictionary and add numeric labels as a new column in dataframe
    unique_labs = np.unique(tktdata[predictedcolumn])
    labels_index={}  # dictionary mapping label name to numeric id
    
    for lab in unique_labs:
        label_id = len(labels_index)
        labels_index[lab] = label_id
    
    tktdata['Codes'] = list(map(lambda x: labels_index[x], tktdata[predictedcolumn]))
    print ('unique class:',unique_labs.shape)
    
    
    # In[7]:
    
    
    # Save Label Index to file.
    
    
    save_obj(labels_index,'savedmodels/'+sl_name+'_'+ver+'_labels_dict.pickle') #sanjukta
    
    
    # In[8]:
    
    
    # Variable initialization
    NUM_CLASSES = len(tktdata[predictedcolumn].value_counts())
    print('Number of different Category values:',NUM_CLASSES)
    MAX_SEQUENCE_LENGTH = 40 #40
    MAX_NUM_WORDS =50000
    EMBEDDING_DIM = 150 #sanjukta-
    VALIDATION_SPLIT = 0.20
    
    
    # In[9]:
    
    
    #Exclude stopwords as per ur need
    stopwordlist = stopwords.words('english')
    stopwordlist.append('would')
    not_stopwords = {'not','up','down','on','off','above','below','between','dear','team','regards'}
    final_stop_words = [word for word in stopwordlist if word not in not_stopwords]
    
    
    # In[10]:
    
    
    
    
    # In[11]:
    
    st.session_state['si_status_placeholder'].warning("Extracting knowledge from Base Model...")
    # Call preprocessing functions
    tktdata['TICKET_PROBLEM_TKTNOTE_ORIG'] =  tktdata[traincolumn]
    tktdata[traincolumn]=list(map(replaceNumber,tktdata[traincolumn]))
    #tktdata[traincolumn]=list(map(replacemultipleNembor,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replaceINC,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replaceREQ,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replaceURL,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replaceEmail,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replacePO,tktdata[traincolumn]))
    tktdata[traincolumn]=list(map(replaceothers,tktdata[traincolumn]))
    #call stopword removal at last as it ll also lowercase the text
    tktdata[traincolumn]=list(map(replacestopword,tktdata[traincolumn]))
    
    
    # In[12]:
    
    
    #Find the maxlength of the list
    MAX_LENGTH = 0        
    for eachSentence in tktdata[traincolumn]:
        wordCount = len(re.findall(r'\w+', eachSentence))
        if wordCount > MAX_LENGTH:
            MAX_LENGTH = wordCount
    print ('MAX_LENGTH of sentence:', MAX_LENGTH)
    
    
    # In[13]:
    
    
    texts = tktdata[traincolumn].str.replace("'","")
    labels = tktdata['Codes']
    
    
    # In[14]:
    
    
    #Load back Tokenizer
    st.session_state['si_status_placeholder'].warning("Model Building...")
    with open('savedmodels/base_V11_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    print(str(datetime.now())[:19]+'-> '+'Tokenizer Loaded Successfully.')
    
    
    # In[15]:
    
    
    # Encoding Text
    sample_rec = tktdata[traincolumn] 
    seq = tokenizer.texts_to_sequences(sample_rec)
    data = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    
    
    # In[16]:
    
    
    # Encode the labels
    labels = to_categorical(np.asarray(labels))
    
    print('Shape of label tensor:', labels.shape)
    
    
    # In[17]:
    
    
    # split the data into a training set and a validation set
    seed = 1234
    np.random.seed(seed)
    
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    
    pre_trained_model = load_model('savedmodels/'+'best_model.hdf5')
    
    
    
    # In[19]:
    
    
    # print all the layers of base model
    print(pre_trained_model.layers)
    
    
    
    
    # In[21]:
    
    
    # Exclude the last layer and keep the remaining layers and save as a reduced model
    pre_trained_model_new=Model(pre_trained_model.layers[0].input,pre_trained_model.layers[-2].output)
    
    #Verify the saved reduced model
    print(pre_trained_model_new.layers)
    
    
    # In[22]:
    
    
    #Take the output from pretrained layer
    x1 = pre_trained_model.layers[2].output
    # Add one LSTM layer in the pretrained layer
    x2 = LSTM(256,dropout=0.5,recurrent_dropout=0.5,name='new_LSTM')(x1)
    #Add last layer to the reduced layer
    layr = Dense(NUM_CLASSES,activation='softmax')(x2)
    
    
    # In[23]:
    
    
    new_model = Model(
        inputs=pre_trained_model_new.input,outputs=layr)
    
    
    
    new_model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['acc',tfa.metrics.F1Score(num_classes=NUM_CLASSES, average="micro")])
    
    
    new_model.summary()
    
    
    # In[28]:
    
    
    for ix in range(len(new_model.layers)):
        print(ix,new_model.layers[ix])
    
    
    # In[29]:
    
    
    startt = datetime.now()
    #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('./savedmodels/'+'superimposed_model.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    #callbacklist=[mc,es]
    callbacklist=[mc]
    num_epochs=3
    train_history = new_model.fit(x_train, y_train,
              batch_size=64,
              epochs=num_epochs,
              validation_split=0.2,
              verbose = 2,
              callbacks=callbacklist)
    # return train_history
    endt = datetime.now()
    
    pred = new_model.predict(data)
    res_prob=list(map(max,pred.tolist()))
    
    
    res=list(map(np.argmax,pred.tolist()))
    #print(res)
    
    
    # In[31]:
    
    
    
    
    labels_pred=list(map(get_label,res))
    
    
    # In[32]:
    
    
    tktdata['Category_Predicted'] = labels_pred
    tktdata['Confidence_Probability'] = list(map(lambda x:round(x*100,2),res_prob))
    
    tktdata['ProcessType'] = 'ML'
    tktdata['keyword_used'] = ''
    
    
    print(str(datetime.now())[:19]+'-> '+'Save predictions to file.')
    
        
    #### Predict Target for Unseen Data
    #### Predict Target for Unseen Data
    
    print(str(datetime.now())[:19]+'-> '+'Get class probabilities for each input..')
    pred = new_model.predict(data)
    res_prob=list(map(max,pred.tolist()))
    
    print(str(datetime.now())[:19]+'-> '+'Get class label index for each input..')
    res=list(map(np.argmax,pred.tolist()))
    #print(res)
    
    st.session_state['si_status_placeholder'].warning("Transfer Learning completed...")
    # In[ ]:
    ## plotting
    eval = new_model.evaluate(x_val, y_val)
    st.session_state['si_status_placeholder'].warning("Transfer Learning Completed...")
    print('Metrics for **'+sl_name+'** Service Line\n')
    print('Accuracy of Train Data     :',round(train_history.history['acc'][-1]*100,2),'%')
    print('Accuracy of Val Data       :',round(train_history.history['val_acc'][-1]*100,2),'%')
    print('F1 Score of Train Data     :',round(train_history.history['f1_score'][-1]*100,2),'%')
    print('F1 Score of Val Data       :',round(train_history.history['val_f1_score'][-1]*100,2),'%')
    #print ('F1 Score of Val Data       :',metrics.val_f1s[-1]*100,'%')
    #print ('Recall Score of Val Data   :',metrics.val_recalls[-1].round(4)*100,'%')
    #print ('Precision Score of Val Data:',metrics.val_precisions[-1].round(4)*100,'%')
    st.session_state['si_status_placeholder'].success("Training Finished.")
    st.session_state['si_train_header'].subheader('Training Result')
    my_expander = st.session_state['si_train_result'].expander(label='Training Summary',expanded=True)
    with my_expander:  
     #train_result=pd.DataFrame({'Accuracy of Train Data':str(round(100,2))+'%','Accuracy of Val Data':str(round(100,2))+'%','F1 Score of Train Data':str(round(100,2))+'%','F1 Score of Val Data':str(round(100,2))+'%'},index=[1])
        
     train_result=pd.DataFrame({'Accuracy of Train Data':str(round(train_history.history['acc'][-1]*100,2))+'%','Accuracy of Val Data':str(round(train_history.history['val_acc'][-1]*100,2))+'%','F1 Score of Train Data':str(round(train_history.history['f1_score'][-1]*100,2))+'%','F1 Score of Val Data':str(round(train_history.history['val_f1_score'][-1]*100,2))+'%'},index=[1])
     st.write(train_result)
    
    
    #
    #st.session_state['predict_df'].write(df)
    st.session_state['si_user_data']=None
    #==========Plot graph
    #========================================Plot Loss and accuracy===========
    accuracy = train_history.history['acc']
    val_accuracy = train_history.history['val_acc']
    
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    save_obj(accuracy,'./report/accuracy.pickle') #sanjukta
    epochs = range(len(accuracy))
    st.session_state['si_status_placeholder'].warning("Transfered Model is ready...")
    
    
    #plt.figure(figsize=(14,8))
    fig, ax = plt.subplots()
    print("epochs:",epochs,accuracy)
    # print("Plotting 1...")
    plt.minorticks_off()
    accuracy = list(map(lambda item: item * 100, accuracy))
    accuracy=[round(num, 1) for num in accuracy]
    val_accuracy = list(map(lambda item: item * 100, val_accuracy))
    val_accuracy=[round(num, 1) for num in val_accuracy]
    ax.plot(list(epochs),accuracy, 'bo-', label='Training')
    ax.plot(list(epochs),val_accuracy, 'co-', label='Validation')
    bins=num_epochs*3/4
    bins=int(bins)
    ax.set_title('Training and validation accuracy')
    #plt.xticks(epochs,[1,5,10])
    print("bins:",bins)
    plt.locator_params(axis='x', nbins=bins)
    yticks=list(np.arange(0, 110, 10))
    yticks_str=[str(d)+'%' for d in yticks]
    plt.yticks(yticks,yticks_str)
    ax.legend(loc="upper left")
    ax.spines['right'].set_color("None")
    ax.spines['top'].set_color("None")
    ax.spines['right'].set_color("None")
    ax.spines['right'].set_color("None")
    ax.set_xlabel("Epoch -------->")
    ax.set_ylabel("Accuracy -------->")
    
    st.session_state['si_status_placeholder'].warning("Plotting Is In Progress...")
    new_model.save('saved_models/3')
    
    plt.savefig('transfer_graph.png')
    
    fig, ax = plt.subplots()
    print("epochs:",epochs,accuracy)
       
    plt.minorticks_off()
    accuracy = list(map(lambda item: item * 100, accuracy))
    accuracy=[round(num, 1) for num in accuracy]
    val_accuracy = list(map(lambda item: item * 100, val_accuracy))
    val_accuracy=[round(num, 1) for num in val_accuracy]
    ax.plot(list(epochs),accuracy, 'bo-', label='Training')
    ax.plot(list(epochs),val_accuracy, 'co-', label='Validation')
    bins=num_epochs*3/4
    bins=int(bins)
    ax.set_title('Training and validation accuracy')
    #plt.xticks(epochs,[1,5,10])
    print("bins:",bins)
    plt.locator_params(axis='x', nbins=bins)
    yticks=list(np.arange(0, 110, 10))
    yticks_str=[str(d)+'%' for d in yticks]
    plt.yticks(yticks,yticks_str)
    ax.legend(loc="upper left")
    ax.spines['right'].set_color("None")
    ax.spines['top'].set_color("None")
    ax.spines['right'].set_color("None")
    ax.spines['right'].set_color("None")
    ax.set_xlabel("Epoch -------->")
    ax.set_ylabel("Accuracy -------->")
    
    image = Image.open('transfer_graph.png')
    
    new_image = image.resize((900, 400))
    
    st.session_state['si_train_plot'].image(new_image)
    st.session_state['si_status_placeholder'].warning("Plotting Finished.")



