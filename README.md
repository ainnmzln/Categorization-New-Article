![badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# Categorization-New-Articles with LSTM

# 1. Summary 

This project is to categorize unseen  articles into 5 categories namely Sport, Tech, Business, Entertainment and  Politics using LSTM approach.

# 2. Dataset

This projects is trained with  [Heart Attack Analysis & Prediction Dataset](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv).

# 3. Requirements
This project is created using [Google Colab](https://colab.research.google.com/?utm_source=scs-index) as the main IDE. The main frameworks used in this project are Pandas, Numpy, Sklearn, TensorFlow and Tensorboard.

# 4. Methodology
This project contains two .py files. The training and modules files are training.py and modules.py respectively. The flow of the projects are as follows:

## 1. Importing the libraries and dataset

The data are loaded from the dataset and usefull libraries are imported.

## 2. Exploratory data analysis

The datasets is cleaned with necessary step. The HTML tags and uppercase are removed and split. The data is then token.

## 3. LSTM model

A Long Short-Term Memory (LSTM) model is created. 

## 4. Model Prediction and Accuracy

The classification report, confusion matrix and accuracy score of the training are shown below. Next, the graph is plotted using Tensorboard.

![Report](https://github.com/ainnmzln/Categorization-New-Article/blob/main/images/report.png)

![](https://github.com/ainnmzln/Categorization-New-Article/blob/main/images/2022-05-19%20(2).png)

