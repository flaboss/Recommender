''' Simple movie recommender
Flavio Bossolan

Dataset and description:
http://files.grouplens.org/datasets/movielens/
http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
'''
#%% import libraries and data
import pandas as pd
import numpy as np
import os 

os.chdir('/home/flavio/Dengue_fever/rec_sys_data')

#reading data
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

#import metadata and create one simple dataset
header2 = ['item_id', 'title', 'releaseDT', 'videoReleaseDT', 'IMDB', 'unknown',
          'action', 'adventure', 'animation', 'children', 'comedy', 'crime'
          'documentary', 'drama', 'fantasy', 'noir', 'horror', 'musical',
          'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
meta= pd.read_csv('u.item', sep='|', names=header2, encoding='latin-1', index_col= False)

#%% Data munging
movies= pd.merge(df, meta, on='item_id', how='inner')

data= movies[['user_id', 'item_id', 'rating', 'title']]
#create a matrix based on the data
matrix= data.pivot_table(index='user_id', columns='title', values='rating')

#%% Functions for Recommending Movies
def pearsonR(s1, s2):
    s1_c = s1-s1.mean()
    s2_c= s2-s2.mean()
    return np.sum(s1_c*s2_c) / np.sqrt(np.sum(s1_c**2)* np.sum(s2_c**2))

def recommend(movie, M, n):
    reviews=[]
    for title in M.columns:
        if title == movie:
            continue
        cor= pearsonR(M[movie], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
            
    reviews.sort(key= lambda tup: tup[1], reverse=True)
    return reviews[:n]

#%% Recommendations

# movies watched
watched= ['Blue Chips (1994)', 'Sunset Park (1996)', 'Hercules (1997)']

#Recommendation for Hercules
recs= recommend('Hercules (1997)', matrix, 10)
recs

#%% Trimmed Rec List: Filtering out the movies I have already watched:

#now filter the recommendations that the user already watched
# and build a new recommendation list:
trimmed_rec_list = [r for r in recs if r[0] not in watched]

#see how it differs from the first recommendation list      
trimmed_rec_list

'''
I have a similar implementation on:
https://www.kaggle.com/flaviobossolan/simple-efficient-movie-recommender/
'''