
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data=pd.read_csv(r'/Users/utkarsh/Python and ML/spam.csv',encoding='latin-1')

data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

data.rename(columns={'v1':'Category','v2':'Email'},inplace=True)

data=data.drop_duplicates()

label_encoder=LabelEncoder() 
data['Category']=label_encoder.fit_transform(data['Category'])

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re 

nltk.download('stopwords')  
nltk.download('punkt') 
nltk.download('wordnet') 


def preprocess_text(text):
    if not isinstance(text, str):
        return ''

    text=text.lower()

    text=re.sub(r"[^a-zA-Z0-9\s]","",text)
    
    text=word_tokenize(text)

    stop_words=set(stopwords.words('English')) 
    text=[word for word in text if word not in stop_words]
    
    lemmatizer=WordNetLemmatizer() 
    text=[lemmatizer.lemmatize(word) for word in text]

    text=' '.join(text)
    
    return text

data['Email']=data['Email'].apply(preprocess_text)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()

x=vectorizer.fit_transform(data['Email'])
y=data['Category']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

model=LogisticRegression()
model.fit(x_train,y_train)

import pickle
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
