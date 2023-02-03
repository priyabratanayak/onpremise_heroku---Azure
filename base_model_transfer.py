#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
'''


# In[2]:


'''
from google.colab import drive
drive.mount('/content/drive')
'''


# In[3]:


# Load required libraries
import numpy as np # For mathematical calculations
import pandas as pd # For Dtaa frames

from PIL import Image
from datetime import datetime

import pickle
import re,os

import matplotlib.pyplot as plt
import warnings

import pickle
import streamlit as st
# Define preprocessing functions
#os.chdir(r'C:\Users\biswa\OneDrive\Documents\Python Tutorial\Streamlite\neurohack')
from userValidation import SigninDetails

reversed_dictionary=None
final_stop_words=None

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

# Save Label Index to file.
def save_obj(obj,name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def get_summary(x):
    text_words = []
    for num in x:
        if num != 0:
            text_words.append(' ')
            text_words.append(reversed_dictionary[num])
    return ''.join(text_words)



 
def format_func(value):
    
        return r"{0}%".format(value)
   

def training_model():
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords')
    import tensorflow as tf
    import tensorflow_addons as tfa
    from tensorflow.keras.preprocessing.text import Tokenizer # For text tokenization
    from tensorflow.keras.preprocessing.sequence import pad_sequences # For padding the text sequences
    from tensorflow.keras.utils import to_categorical # To do one hot encoding of the numeric lable indexes
    from tensorflow.keras.layers import Input,Dense,Embedding,LSTM # NN Layers
    from tensorflow.keras.models import Model,load_model # 
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
    #from tensorflow.metrics import f1_score,recall_score,precision_score
    from tensorflow.keras.metrics import Recall,Precision
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras import optimizers
    global final_stop_words,reversed_dictionary
    class Metrics(Callback):
        def __init__(self, monitor='val_acc', value=91, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
            '''
       def on_train_begin(self, logs={}):
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
            self.monitor = monitor
            self.value = value
            self.verbose = verbose
            '''
    starttime = datetime.now()
    st.session_state['status_placeholder'].warning("Fetching Data...")
    #print('Starting time:',starttime)
    sl_name = 'base' #'Network' 'SWServicesDB' ##for model file name--sanjukta
    ver = 'V11' ##for model file name--sanjukta
    traincolumn="Short description"
    predictedcolumn="Assignment group"
    glovevectorfile='./glove/glove_150.txt'
    
    
    oddlogic_Prediction=SigninDetails("mongodb+srv://oddlogic:oddlogic@cluster0.8qa4jjw.mongodb.net/?retryWrites=true&w=majority","oddlogic")
    oddlogic_Prediction_glove=SigninDetails('mongodb+srv://oddlogic:oddlogic@cluster0.h52iyb6.mongodb.net/?retryWrites=true&w=majority',"oddlogic")
    oddlogic_Prediction.create_Collection("Input")   
    
  
    
    tktdata=None 
    data_from_db = oddlogic_Prediction.collection_dict["Input"].find({},{'_id':0})
    tktdata=pd.DataFrame.from_dict(data_from_db)
    unique_labs = np.unique(tktdata[predictedcolumn])
    labels_index={}  # dictionary mapping label name to numeric id
    
    for lab in unique_labs:
        label_id = len(labels_index)
        labels_index[lab] = label_id
    #print(unique_labs.shape)
    #print(tktdata.columns,tktdata.shape)
    tktdata=tktdata.head(1000)
    #print("tktdata:",tktdata.head(3))
    st.session_state['status_placeholder'].warning("Data Fetched successfully")
    save_obj(labels_index,'./savedmodels/'+sl_name+'_'+ver+'_labels_dict.pickle') #sanjukta
    
    
   
    
    
    # Variable initialization
    NUM_CLASSES = len(tktdata[predictedcolumn].value_counts())
    #print('Number of different Category values:',NUM_CLASSES)
    MAX_SEQUENCE_LENGTH = 40 #40
    MAX_NUM_WORDS =50000
    EMBEDDING_DIM = 150 #sanjukta-
    VALIDATION_SPLIT = 0.20
    
    
    # In[12]:
    
    
    #Exclude stopwords as per ur need
    stopwordlist = stopwords.words('english')
    stopwordlist.append('would')
    not_stopwords = {'not','up','down','on','off','above','below','between','dear','team','regards'}
    final_stop_words = [word for word in stopwordlist if word not in not_stopwords]
    
    
    # In[13]:
    
    
    
    
    # In[14]:
    
    
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
    
    
  
    
    # In[16]:
    
    
    #Find the maxlength of the list
    MAX_LENGTH = 0        
    for eachSentence in tktdata[traincolumn]:
        wordCount = len(re.findall(r'\w+', eachSentence))
        if wordCount > MAX_LENGTH:
            MAX_LENGTH = wordCount
    print ('MAX_LENGTH of sentence:', MAX_LENGTH)
    
    
    # In[17]:
    
    
    texts = tktdata[traincolumn].str.replace("'","")
    labels = tktdata['Codes']
    
    print(labels.shape)
    # In[18]:
    
    
    # Tokenize the corpus into tokens..converts to default lower case 
    
    # Do not remove '_' from Summary. Default filter : '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    #tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n79')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    #print ('tokens are',word_index)
    # Reversed dictionary useful to convert predicted sequences to words during model prediction verification.
    reversed_dictionary = dict(zip(word_index.values(), word_index.keys()))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    # In[19]:
    
    
    # ### Saving model and loading it when needed
    
    
    #Save Tokenizer
    with open('./savedmodels/'+sl_name+'_'+ver+'_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #print("Saved tokenizer to disk")
    
    # Save Model
    #model.save('./savedmodels/'+sl_name+'_'+ver+'_model.hdf5')    
    ##print("Saved model to disk")
    
    
    # In[20]:
    
    
    # Recalculate summary
    
    
    
    # In[21]:
    
    
    tktdata['traincolumn']=list(map(get_summary,sequences))
    
    tktdata.tail(10)
    
    
    # In[22]:
    
    labels_b=labels.copy()
    print(labels.shape)
    # Encode the labels
    #labels = to_categorical(np.asarray(labels))
    labels = to_categorical(labels)
    print(labels.shape)
    # In[23]:
    
    
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
    
    print(x_train.shape)
    print("y_train:",y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    
    
    # In[24]:
    
    
    ## Save Train and Test datasets to files.
     
    ## Save Train and Test datasets to files.
    
    tktdata_shuff = tktdata.iloc[indices,:]
    train_df =  tktdata_shuff[:-num_validation_samples]
    test_df =  tktdata_shuff[-num_validation_samples:]
    if st.session_state['chkbox_csv_file']:
        train_df=st.session_state['user_data']
    else:
        train_df.to_excel('./input/'+sl_name+'_'+ver+'_Train.xlsx', index=False) #sanjukta
        
    if st.session_state['data_external']:
        try:
            train_df=pd.read_csv( st.session_state['data_external'])
        except:
            train_df.to_excel('./input/'+sl_name+'_'+ver+'_Train.xlsx', index=False) #sanjukta
            
    
    
   #test_df.to_excel('./savedmodels/'+sl_name+'_'+ver+'_Test.xlsx', index=False) #sanjukta
    
    
    # In[25]:
    
    
    
    #metrics = Metrics()
    #add paramater to stop training when val_acc reaches a value
    metrics = Metrics(monitor='val_acc', value=.94, verbose=1)
    
    
    # In[26]:
    
    
    # Put Glove word and corresponding vectors into a dict
    #print('Indexing word vectors.')
    
   
    
    Collection_Credentials="glove"
    oddlogic_Prediction_glove.create_Collection(Collection_Credentials)
    
    
    
    output=oddlogic_Prediction_glove.download_gloveFile('glove_150')
    
    embeddings_index={}
    for line in output.decode('UTF-8') .splitlines():
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print (line)
        embeddings_index[word] = coefs
    

    st.session_state['status_placeholder'].warning('Fetching Embedding File...')
    print("Dict embeddings_index:",len(embeddings_index)) 
    # #### Prepare Embedding Matrix based on Glove Vectors
    
    #print('Preparing embedding matrix.')
    
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    j=0
    k=0
    n=0
    nd = []
    for word, i in word_index.items():
        k=i
        ##print (word,i)
        #if i >= MAX_NUM_WORDS:
         #   continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:        #if word found
            ##print ("found:",word,i)
            embedding_matrix[i] = embedding_vector
            n=n+1
        else:
            ##print ("not found",word,i)
            nd.append(word)
            j=j+1
    #print ('Total no of  iteration/corpus wordcount:',k)       
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = True(Default) so as to keep the embeddings trained with current data.
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable = True)        
    
    embedding_matrix.shape
    
    #print ("Found Wordcount:",n)
    #print ("not found word count",j)
    #print ("Words present in corpus but not in glove",nd)
    '''new glove Total no of  iteration/corpus wordcount: 2450
    Found Wordcount: 2181
    not found word count 269
    
    '''
    
    
    # In[28]:
    
    
    embeddings_index.get("'private")
    
    
    # In[29]:
    
    st.session_state['status_placeholder'].warning('Embedding File Fetched')
    # #### Build Model
    
    #print('Buiding model.')
    
    opt=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    reg = tf.keras.regularizers.l1_l2(l1=0,l2=0.001 )
    init_mode="HeUniform"
    '''
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    output_1 = BatchNormalization()(embedded_sequences)
    output_2=Dense(512,kernel_regularizer=reg,kernel_initializer=init_mode,activation='relu')(output_1)
    output_3= BatchNormalization()(output_2)
    x1 = LSTM(256,dropout=0.3,recurrent_dropout=0.4,return_sequences= True)(output_3)
    output_4= BatchNormalization()(x1)
    x2 = LSTM(128,dropout=0.3,recurrent_dropout=0.4)(output_4)
    output_5= BatchNormalization()(x2)
    preds = Dense(NUM_CLASSES, activation='softmax')(output_5)
    
    '''
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x1 = LSTM(256,dropout=0.5,recurrent_dropout=0.5,return_sequences= True)(embedded_sequences)
    x2 = LSTM(256,dropout=0.5,recurrent_dropout=0.5)(x1)
    preds = Dense(NUM_CLASSES, activation='softmax')(x2)
    
    model = Model(sequence_input,preds)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc',tfa.metrics.F1Score(num_classes=NUM_CLASSES, average="micro")])

    
    model.summary()
    
    
    
    # simple early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint('./savedmodels/'+'best_model.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    #callbacklist=[metrics,mc]
    callbacklist=[mc]
    #Fit model
    startt = datetime.now()
    print(x_train.shape,y_train.shape)
    num_epochs=5
    st.session_state['status_placeholder'].warning("Training Is In Progress...")
    train_history = model.fit(x_train, y_train,
              batch_size=64,
              epochs=num_epochs,
              validation_split=0.2,
              verbose = 2,
              callbacks=callbacklist)
    
    endt = datetime.now()
    #print('\n\nTotal time taken for training:',endt - startt)
    
    
    # In[32]:
    
    
    # #### Accuracy and F1_Score on Validation data
    
    #print('Metrics for **'+sl_name+'** Service Line\n')
    eval = model.evaluate(x_val, y_val)
    save_obj(metrics,'./savedmodels/metrics_'+sl_name+'_'+ver+'.pickle') #sanjukta
    
    # #### Accuracy and F1_Score on Validation data

    
    eval = model.evaluate(x_val, y_val)
    print('Metrics for **'+sl_name+'** Service Line\n')
    print('Accuracy of Train Data     :',round(train_history.history['acc'][-1]*100,2),'%')
    print('Accuracy of Val Data       :',round(train_history.history['val_acc'][-1]*100,2),'%')
    print('F1 Score of Train Data     :',round(train_history.history['f1_score'][-1]*100,2),'%')
    print('F1 Score of Val Data       :',round(train_history.history['val_f1_score'][-1]*100,2),'%')
    #print ('F1 Score of Val Data       :',metrics.val_f1s[-1]*100,'%')
    #print ('Recall Score of Val Data   :',metrics.val_recalls[-1].round(4)*100,'%')
    #print ('Precision Score of Val Data:',metrics.val_precisions[-1].round(4)*100,'%')
    st.session_state['status_placeholder'].success("Training Finished.")
    st.session_state['train_header'].subheader('Training Result')
    my_expander = st.session_state['train_result'].expander(label='Training Summary',expanded=True)
    with my_expander:  
     #train_result=pd.DataFrame({'Accuracy of Train Data':str(round(100,2))+'%','Accuracy of Val Data':str(round(100,2))+'%','F1 Score of Train Data':str(round(100,2))+'%','F1 Score of Val Data':str(round(100,2))+'%'},index=[1])
        
     train_result=pd.DataFrame({'Accuracy of Train Data':str(round(train_history.history['acc'][-1]*100,2))+'%','Accuracy of Val Data':str(round(train_history.history['val_acc'][-1]*100,2))+'%','F1 Score of Train Data':str(round(train_history.history['f1_score'][-1]*100,2))+'%','F1 Score of Val Data':str(round(train_history.history['val_f1_score'][-1]*100,2))+'%'},index=[1])
     st.write(train_result)
    
    
    #
    #st.session_state['predict_df'].write(df)
    st.session_state['user_data']=None
    #==========Plot graph
    #========================================Plot Loss and accuracy===========
    accuracy = train_history.history['acc']
    val_accuracy = train_history.history['val_acc']
    
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    save_obj(accuracy,'./report/accuracy.pickle') #sanjukta
    epochs = range(len(accuracy))
    
    
    
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
    
    
    st.session_state['status_placeholder'].warning("Plotting Is In Progress...")
    model.save('saved_models/1')
    
    plt.savefig('training_graph.png')
    
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
    
    image = Image.open('training_graph.png')
    
    new_image = image.resize((900, 400))
    
    st.session_state['train_plot'].image(new_image)
    st.session_state['status_placeholder'].warning("Plotting Finished.")
if __name__=="__main__":
    training_model()

