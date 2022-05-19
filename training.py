# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:41:28 2022

@author: ainnmzln
"""

import os,datetime
import pandas as pd
import numpy as np
from module import ExploratoryDataAnalysis,ModelCreation,ModelEvaluation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping


#%% Saved model
URL=('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
PATH=os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','model.h5')
TOKENIZER_JSON_PATH=os.path.join(os.getcwd(),'saved_models','tokenizer.json')

#%% Step 1. Load data

df=pd.read_csv(URL)
category = df['category']
text = df['text']
text_dummy =text.copy()

#%% Step 2. EDA

eda=ExploratoryDataAnalysis()

# Remove tag, uppercase, token and pad sequences the text
text_dummy=eda.remove_tags(text_dummy)
text_dummy=eda.remove_uppercase(text_dummy)
text_dummy=eda.sentiment_token(text_dummy,
                               token_save_path=TOKENIZER_JSON_PATH,
                               prt=True)
text_dummy=eda.sentiment_pad_sequences(text_dummy)

# Encode the category 
category_enc=eda.encoder(category)

# Test and train split dataset

# Train & test data
X_train, X_test, y_train, y_test = train_test_split(text_dummy, 
                                                    category_enc, 
                                                    test_size=0.3, 
                                                    random_state=123)

# Expand the dimension to fit into model
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)


#%% Step 3. Model creation

nb_categories = len(category.unique())
mc = ModelCreation()
model = mc.lstm_layer(10000, nb_categories)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics='acc')

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=3)

log_dir=os.path.join(PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir)

hist = model.fit(X_train, y_train, epochs=20,
                 validation_data=(X_test,y_test), 
                 callbacks=[early_stopping_callback,
                            tensorboard_callback])

#%% Step 4. Model Evaluation

predicted = np.empty([len(X_test),5])

for index, test in enumerate(X_test):
    predicted[index,:] = model.predict(np.expand_dims(test, axis=0))

y_pred = np.argmax(predicted, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

#%% Step 5. Model Deployment
model.save(MODEL_SAVE_PATH)