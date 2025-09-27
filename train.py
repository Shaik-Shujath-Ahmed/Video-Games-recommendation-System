#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sklearn


# In[51]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
d=pd.read_csv(r"video_game_reviews.csv")
d.dropna(inplace=True)
d.drop_duplicates(inplace=True)


# In[52]:


d.drop(axis=1,columns=['Requires Special Device', 'Developer', 'Publisher','Game Length (Hours)', 'Graphics Quality',
       'Soundtrack Quality', 'Story Quality',
       'Min Number of Players'],inplace=True)


# In[53]:



bins = [10, 20, 30, 40, 45, 50]

labels = ['Very Low Rating', 'Low Rating', 'Medium Rating', 'High Rating', 'Very High Rating']

d['User Rating'] = pd.cut(
    d['User Rating'],
    bins=bins,
    labels=labels,
    include_lowest=True)


# In[113]:



X = d.drop(columns=["Game Title"])
y = d["Game Title"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
ordinal_features = ['User Rating', 'Age Group Targeted']
ordinal_categories = [
    ['Very Low Rating','Low Rating','Medium Rating','High Rating','Very High Rating'],
    ['Kids', 'Teens', 'All Ages', 'Adults']
]
nominal_features = ['Platform', 'Genre', 'Multiplayer', 'Game Mode']
numeric_features = ['Price', 'Release Year']

# Column transformer
preprocessor = ColumnTransformer(transformers=[
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
    ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
    ('passthrough', 'passthrough', numeric_features)
])

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', GaussianNB())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=1, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save pipeline and label encoder
joblib.dump(pipeline, "pipelines.pkl")
joblib.dump(label_encoder, "label_encoders.pkl")



# In[114]:


# In[ ]:




