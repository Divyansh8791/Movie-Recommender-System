import numpy as np
import pandas as pd
import ast




####################################################################################### DATA PREPROCESSING PART ###########################################################

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
# print(movies.head(1))
# print(credits.head(1)['cast'].values)

############################################## merging the data ####

movies = movies.merge(credits , on='title')

############################################## remove the useless columns ####
# required columns are : genres , id , keywords , title , overview , release_date(can be) , cast , crew 
movies = movies[['movie_id' , 'title' , 'overview' , 'genres' , 'keywords' , 'cast' , 'crew']]
# print(movies.head(1))

############################################## now check the empty data in all cols

# print(movies.isnull().sum())
movies.dropna(inplace=True)
# print(movies.isnull().sum())

# ast.literal_eval   # but to work this first change the string list to actual list . by a module ast . ast.literal_eval

# print(movies.iloc[0].genres) # it is in : [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"} : form.
# chnage it into : ['Action' , 'Adventure' , 'Fantacy' , 'SciFi'] : for this create a function . 
def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    
movies['genres']= movies['genres'].apply(convert)
movies['keywords']= movies['keywords'].apply(convert)

def convert2(obj):
    L =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3 :
            L.append(i['name'])
            counter+=1
        else :
            break    
    return L    
movies['cast'] = movies['cast'].apply(convert2) 

def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director' :
              L.append(i['name'])
              break
    return L   

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x : x.split())

# now from all the strings we have to remove a " " . we will do it with a very simple function .
movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" " ,"") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" " ,"") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" " ,"") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" " ,"") for i in x])

# now create a tag column which consist of all of these colmns.
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# print(movies['tags'].head(1))  

# now make a new data frame which only consist : movie_id , title , tags : cols
new_df = movies[['movie_id' , 'title' , 'tags']]
# print(new_df)
 
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
# print(new_df['tags'].head())
# # all converted into string formate.
# now change it in lowercase.
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# print(new_df.head())

############################################################################################### DATA VECTORIZATION ######################


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 , stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# print(cv.get_feature_names_out())

# now the problem is that the same names like : actor , actors ; loving , loved.....: are appearing more times . so i should remove them
# for that i should apply stamming on them . 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']= new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 , stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# now we have to calculate the distance for each movie pair.
# we will calcuate the cosine distance not the euclidean distance .
# because in high dimention euclidean distance fails.
# cosine distance nothing but the angle betwen each movie pair.
# less the angle closest the movie.

# for this there is a function in scikit learn called as :  cosine similarity.

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# print(sim)

def recommend(movie):
    movie_idx = new_df[new_df['title']==movie].index[0]  # it will give us the index of our movie.
    distances = similarity[movie_idx]
    movie_list = sorted(list(enumerate(distances)) , reverse=True , key=lambda x : x[1])[0:5]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        # print(i[0])

recommend('Batman Begins')        

import pickle
# pickle.dump(new_df , open('movies.pkl' , 'wb'))
# but the pandas dataframe can not be supplied here.
# so rather than data fram supply dictionary.

pickle.dump(new_df.to_dict() , open('movie_dict.pkl' , 'wb'))
pickle.dump(similarity , open('similarity.pkl' , 'wb'))