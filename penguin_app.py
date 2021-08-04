import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

#Dataframe
df = pd.read_csv('penguin.csv')
df = df.dropna()

#Mapping
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
df['sex'] = df['sex'].map({'Male':0,'Female':1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})

#Models
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

#Prediction Function
@st.cache()
def prediction(model,island,bl,bd,fl,bm,sex):
  predict=model.predict([[island,bl,bd,fl,bm,sex]])
  predicted=predict[0]
  if predicted==0:
    return 'Adelie'
  elif predicted==1:
    return 'Chinstrap'
  else:
  	return 'Gentoo'

#Sidebar
st.sidebar.title('Penguin Prediction Model')
bl=st.sidebar.slider('Bill Length: ',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bd=st.sidebar.slider('Bill Depth: ',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
fl=st.sidebar.slider('Flipper Length: ',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
bm=st.sidebar.slider('Body Mass: ',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))
m=st.sidebar.selectbox('Model',('SVC','Logistic Regression','Random Forest Classifier'))
s=st.sidebar.selectbox('Sex',('Male','Female'))
i=st.sidebar.selectbox('Island',('Biscoe','Dream','Torgerson'))

#Button
island={'Biscoe':0,'Dream':1,'Torgersen':2}
sex={'Male':0,'Female':1}
if st.sidebar.button('Predict'):

  #SVC
  if m=='SVC':
    pred=prediction(svc_model,island[i],bl,bd,fl,bm,sex[s])
    score=svc_model.score(X_train,y_train)
  
  #LogisticRegression
  elif m=='Logistic Regression':
    pred=prediction(log_reg,island[i],bl,bd,fl,bm,sex[s])
    score=log_reg.score(X_train,y_train)
  
  #RFC
  else:
    pred=prediction(rf_clf,island[i],bl,bd,fl,bm,sex[s])
    score=rf_clf.score(X_train,y_train)
      
  st.write('The species has been predicted as: ',pred)
  st.write('The accuracy of the prediction is: ',score)
