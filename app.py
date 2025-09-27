#!/usr/bin/env python
# coding: utf-8

# In[50]:



import streamlit as st
import joblib

# In[115]:
import pandas as pd
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


st.set_page_config(
    page_title="VGRS")
st.markdown("""
<style>
/* Overall App Background */
body, .stApp {
    background: linear-gradient(to bottom right, #f8fafd, #eef2fb);  /* light pastel background */
    color: #222;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Title with gradient neon text */
h1 {
    background: linear-gradient(90deg, #00f0ff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5em;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.6), 0 0 16px rgba(255, 0, 255, 0.4);
}

/* Styled Selects, Sliders, Multiselects */
.stSelectbox > div, .stSlider, .stMultiSelect > div {
    background-color: #ffffff;
    border: 2px solid #00f0ff;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
    transition: box-shadow 0.3s ease;
}

.stSelectbox > div:hover, .stSlider:hover, .stMultiSelect > div:hover {
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.6);
}

/* Neon Button with rainbow glow */
button[kind="primary"] {
    background: linear-gradient(90deg, #00f0ff, #a200ff);
    color: #fff !important;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1.2em;
    box-shadow: 0 0 10px #00f0ff;
    transition: all 0.3s ease;
}

button[kind="primary"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px #a200ff;
}

/* Success prediction box */
.stAlert-success {
    background-color: #ecf9ff !important;
    border-left: 6px solid #00f0ff !important;
    color: #007c91 !important;
    font-weight: bold;
}

/* Table header with shiny colors */
.stDataFrame thead th {
    background: linear-gradient(to right, #00f0ff, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    text-shadow: 0 0 6px rgba(0, 240, 255, 0.5);
}

/* Table rows */
.stDataFrame tbody td {
    background-color: #ffffff !important;
    color: #222 !important;
}

.stDataFrame tbody tr:hover td {
    background-color: #f0faff !important;
    box-shadow: inset 0 0 10px #00f0ff;
}

/* Expander */
.stExpanderHeader {
    color: #00f0ff !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# In[116]:


# Load trained pipeline and label encoder
pipeline = joblib.load("pipelines.pkl")
label_encoder = joblib.load("label_encoders.pkl")

st.title("🎮 Video Game Recommendation System")

# Step-by-step feature selection
release_years = sorted(d['Release Year'].dropna().unique())
selected_year = st.selectbox("Select Release Year", release_years)

filtered_df = d[d['Release Year'] == selected_year]

game_modes = filtered_df['Game Mode'].dropna().unique()
selected_game_mode = st.selectbox("Select Game Mode", game_modes)

filtered_df = filtered_df[filtered_df['Game Mode'] == selected_game_mode]

multiplayer_options = filtered_df['Multiplayer'].dropna().unique()
selected_multiplayer = st.selectbox("Select Multiplayer Option", multiplayer_options)

filtered_df = filtered_df[filtered_df['Multiplayer'] == selected_multiplayer]

platforms = filtered_df['Platform'].dropna().unique()
selected_platform = st.selectbox("Select Platform", platforms)

filtered_df = filtered_df[filtered_df['Platform'] == selected_platform]

genres = filtered_df['Genre'].dropna().unique()
selected_genre = st.selectbox("Select Genre", genres)

filtered_df = filtered_df[filtered_df['Genre'] == selected_genre]

age_groups = filtered_df['Age Group Targeted'].dropna().unique()
selected_age_group = st.selectbox("Select Age Group Targeted", age_groups)

filtered_df = filtered_df[filtered_df['Age Group Targeted'] == selected_age_group]

user_ratings = filtered_df['User Rating'].dropna().unique()
selected_user_rating = st.selectbox("Select User Rating", user_ratings)

prices = sorted(filtered_df['Price'].dropna().unique())
selected_price = st.select_slider(
    "Select Price",
    options=prices,
    value=prices[0],
    format_func=lambda x: f"${x:.2f}"
)
filtered_df = filtered_df[(filtered_df['Price'] <= selected_price)&(filtered_df['User Rating']==selected_user_rating)]
# Final input for prediction
input_df = pd.DataFrame([{
    'User Rating': selected_user_rating,
    'Age Group Targeted': selected_age_group,
    'Platform': selected_platform,
    'Genre': selected_genre,
    'Multiplayer': selected_multiplayer,
    'Game Mode': selected_game_mode,
    'Price': selected_price,
    'Release Year': selected_year
}])

# Predict game
if st.button("🎮 Recommend Video Game"):
    prediction = pipeline.predict(input_df)
    predicted_title = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🎯 Recommended Game: **{predicted_title}**")

# Optionally show filtered rows from final selection
with st.expander("🔍 View Games Matching Your Criteria"):
    st.dataframe(filtered_df[["Game Title","Price"]])
st.write("\n")


# In[ ]:





# In[ ]:




