import streamlit as st
import pandas as pd
st.title('Movie Recommender System')
import requests
import pickle 
# but the pandas dataframe can not be supplied here.
# so rather than data fram supply dictionary.
movies_list = pickle.load(open('movie_dict.pkl' , 'rb'))
movies = pd.DataFrame(movies_list)

select_movie_name = st.selectbox(
    'Enter the Movie name  ',
    # ('name') #  but instead this we have to pass our recommended movie names. for that we will use pickel library.
    movies['title'].values
)

similarity = pickle.load(open('similarity.pkl' , 'rb'))

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=4829f296ac8b0af4add06373f35695a9'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']



def recommend(movie):
    movie_idx = movies[movies['title']==movie].index[0]  # it will give us the index of our movie.
    distances = similarity[movie_idx]
    movie_list = sorted(list(enumerate(distances)) , reverse=True , key=lambda x : x[1])[0:5]
     
    recomended_movies = []
    rec_post = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recomended_movies.append(movies.iloc[i[0]].title)
        rec_post.append(fetch_poster(movie_id))
    return recomended_movies, rec_post


if st.button('Recommend'):
    names , posters  = recommend(select_movie_name)
    col1 , col2 , col3 , col4 , col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])


