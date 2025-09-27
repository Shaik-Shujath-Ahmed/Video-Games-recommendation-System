
import streamlit as st
import pandas as pd
import joblib
import pandas as pd
d=pd.read_csv("src/video_game_reviews.csv")
d.dropna(inplace=True)
d.drop_duplicates(inplace=True)
d.drop(axis=1,columns=['Requires Special Device', 'Developer', 'Publisher','Game Length (Hours)', 'Graphics Quality',
       'Soundtrack Quality', 'Story Quality',
       'Min Number of Players'],inplace=True)
bins = [10, 20, 30, 40, 45, 50]

labels = ['Very Low Rating', 'Low Rating', 'Medium Rating', 'High Rating', 'Very High Rating']

d['User Rating'] = pd.cut(
    d['User Rating'],
    bins=bins,
    labels=labels,
    include_lowest=True)
pipeline = joblib.load("src/baggings.pkl")
st.set_page_config(
    page_title="VGRS")


st.title("🎮 Video Game Recommendation System")


release_years = sorted(d['Release Year'].dropna().unique())
selected_year = st.selectbox("Select Game Release Year", release_years)

filtered_df = d[d['Release Year'] == selected_year]
if filtered_df.empty:
    st.warning("No games found for the selected Release Year.")
    st.stop()
with st.expander("🔍 Choose Games Rating Criteria"):
    user_ratings = filtered_df['User Rating'].dropna().unique()
    selected_user_rating = st.selectbox("Select Desired Game Rating", user_ratings)
filtered_df = filtered_df[filtered_df['User Rating'] == selected_user_rating]
if filtered_df.empty:
    st.warning("No games found for the selected User Rating.")
    st.stop()
with st.expander("🔍 Choose Games Mode Criteria"):
    game_modes = filtered_df['Game Mode'].dropna().unique()
    selected_game_mode = st.selectbox(" Select Preferred Game Mode", game_modes)
filtered_df = filtered_df[filtered_df['Game Mode'] == selected_game_mode]
if filtered_df.empty:
    st.warning("No games found for the selected Game Mode.")
    st.stop()
with st.expander("🔍 Choose Multiplayer Support"):
    multiplayer_options = filtered_df['Multiplayer'].dropna().unique()
    selected_multiplayer = st.selectbox("Multiplayer Support", multiplayer_options)
filtered_df = filtered_df[filtered_df['Multiplayer'] == selected_multiplayer]
if filtered_df.empty:
    st.warning("No games found for the selected Multiplayer.")
    st.stop()
with st.expander("🔍 Choose Gaming Platform"):
    platforms = filtered_df['Platform'].dropna().unique()
    selected_platform = st.selectbox("Select Gaming Platform",platforms)
filtered_df = filtered_df[filtered_df['Platform'] == selected_platform]
if filtered_df.empty:
    st.warning("No games found for the selected Platform.")
    st.stop()
with st.expander("🔍 Choose Game Genre"):
    genres = filtered_df['Genre'].dropna().unique()
    selected_genre = st.selectbox("Select Game Genre", genres)
filtered_df = filtered_df[filtered_df['Genre'] == selected_genre]
if filtered_df.empty:
    st.warning("No games found for the selected Genre.")
    st.stop()
with st.expander("🔍 Choose Player Age Category"):
    age_groups = filtered_df['Age Group Targeted'].dropna().unique()
    selected_age_group = st.selectbox("Select Player Age Category",age_groups)
filtered_df = filtered_df[filtered_df['Age Group Targeted'] == selected_age_group]
if filtered_df.empty:
    st.warning("No games found for the selected Age Group Targeted.")
    st.stop()

filtered_df=filtered_df.sort_values(by="Price")
filtered_df=filtered_df[(filtered_df["Game Title"].duplicated())!=True]

prices = sorted(filtered_df['Price'].dropna().unique())

if len(prices)>1:
    selected_price = st.select_slider("Select Price",options=prices,value=prices[0],format_func=lambda x: f"${x:.2f}")
elif len(prices)==1:
    with st.expander("🔍 View The Price of the Game upon your Criteria."):    
        selected_price = prices[0]
        st.info(f"Only one price available: ${selected_price:.2f}")
else:
    st.warning("⚠️ No price options available for the selected criteria.")
    st.stop()

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
recom=None
if st.button("🎮 Click Here To Recommend Video Game"):
    prediction = pipeline.predict(input_df)
    recom=prediction[0]
    st.success(f"🎯 {recom}")
    st.balloons()
    filtered_df = filtered_df[(filtered_df["Game Title"] != recom)&(filtered_df['Price'] <= selected_price)&(filtered_df['User Rating']==selected_user_rating)]
    if len(prices)>1 and filtered_df.empty!=True:
        with st.expander("🔍 View Games Matching Your Criteria"):
            st.dataframe(filtered_df[["Game Title", "Price"]])
            st.markdown("🎉 Thank you for using my Video Game Recommendation System!", unsafe_allow_html=True)
            st.stop()
#filtered_df = filtered_df[(filtered_df["Game Title"] != recom)&(filtered_df['Price'] <= selected_price)&(filtered_df['User Rating']==selected_user_rating)]
    elif selected_price==prices[0] and filtered_df.empty:
        st.markdown("🎉 Thank you for using my Video Game Recommendation System!", unsafe_allow_html=True)
        st.stop()
    else:
        pass
#if selected_price==prices[0] and filtered_df.empty:
    #st.markdown("🎉 Thank you for using my Video Game Recommendation System!", unsafe_allow_html=True)
    #st.stop()

#elif selected_price==prices[0] and (not filtered_df.empty):
    #with st.expander("🔍 View Games Matching Your Criteria"):
        #st.dataframe(filtered_df[["Game Title", "Price"]])
        #st.markdown("🎉 Thank you for using my Video Game Recommendation System!", unsafe_allow_html=True)
#elif len(prices) > 1:
    #with st.expander("🔍 View Games Matching Your Criteria"):
        #st.dataframe(filtered_df[["Game Title", "Price"]])
        #st.markdown("🎉 Thank you for using my Video Game Recommendation System!", unsafe_allow_html=True)
#else:
    #pass
