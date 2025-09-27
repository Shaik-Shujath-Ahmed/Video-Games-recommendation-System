#!/usr/bin/env python
# coding: utf-8

# In[56]:

# In[94]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
d=pd.read_csv(r"video_game_reviews.csv")
d.dropna(inplace=True)
d.drop_duplicates(inplace=True)
d.drop(axis=1,columns=['Requires Special Device', 'Developer', 'Publisher','Game Length (Hours)', 'Graphics Quality',
       'Soundtrack Quality', 'Story Quality',
       'Min Number of Players','User Review Text'],inplace=True)
bins=[10, 20, 30, 40, 45, 50]

values= ['Very Low Rating', 'Low Rating', 'Medium Rating', 'High Rating', 'Very High Rating']

d['User Rating'] = pd.cut(d['User Rating'],bins=bins,labels=values,include_lowest=True)
x= d.drop(columns=["Game Title"])
y = d["Game Title"]
ordinal_features = ['User Rating', 'Age Group Targeted']
ordinal_categories = [['Very Low Rating','Low Rating','Medium Rating','High Rating','Very High Rating'],['Kids', 'Teens', 'All Ages', 'Adults']]
nominal_features = ['Platform', 'Genre', 'Multiplayer', 'Game Mode']
numeric_features = ['Price', 'Release Year']
preprocessor = ColumnTransformer(transformers=[('ord', OrdinalEncoder(categories=ordinal_categories),ordinal_features),('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),('passthrough','passthrough', numeric_features)])
pipeline = Pipeline(steps=[('preprocessing', preprocessor),('classifier', BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=20,bootstrap=True,random_state=29))])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1,random_state=29)
pipeline.fit(x_train,y_train)
joblib.dump(pipeline, "baggings.pkl")


# In[ ]:




