# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:23:08 2022

@author: ainnmzln
"""
import re,json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding,Bidirectional
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    def remove_tags(self,data):
        for index,text in enumerate(data):
            data[index]=re.sub('<.*?', '',text)
            
        return data 
 
    def remove_uppercase(self,data):
        for index, text in enumerate(data):
            data[index] = re.sub("[^a-zA-Z]", " ", text).lower().split()
        
        return data
    def sentiment_token(self,data,token_save_path,num_words=10000,
                        oov_token='<OOV>',prt=False):

        # Tokenizer to vectorize the words
        tokenizer=Tokenizer(num_words=num_words,oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        # To save the tokenizer 
        token_json=tokenizer.to_json()

        with open(token_save_path,'w') as json_file:
            json.dump(token_json,json_file)
        
        # To observe the number of words 
        word_index=tokenizer.word_index
        
        if prt==True:
            print(word_index)
            print(list(word_index.items())[0:10])
            
        # To vectorize the sequence of text
        data=tokenizer.texts_to_sequences(data)
        
        return data 
    
    def sentiment_pad_sequences(self,data):
        
        data=pad_sequences(data,maxlen=200,
              padding='post',truncating='post')
        
        return data
    
    def encoder(self,data):
        ohe=OneHotEncoder(sparse=False)
        data_enc=ohe.fit_transform(np.expand_dims(data,
                               axis=-1))
        return data_enc
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self,num_words,nb_categories,
                   embedding_output=64,
                   nodes=32,dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) 
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))  
        model.summary()
        return model
    
class ModelEvaluation():
    
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))

if __name__=='__main__':
    pass

