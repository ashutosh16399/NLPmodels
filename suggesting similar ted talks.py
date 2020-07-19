# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:30:48 2020

@author: Ashut
"""
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
metadata=pd.read_csv('ted.csv')

def prepro(title):
    pattern = r"\w+_\w+"
    return(re.findall(pattern, title)[0])
    
metadata['url']= metadata['url'].apply(prepro)

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['url']).drop_duplicates()
transcripts=pd.Series(metadata['transcript']).drop_duplicates()

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['url'].iloc[movie_indices]


 
# Generate recommendations
print(get_recommendations('nic_marks_the_happy_planet_index', cosine_sim, indices))