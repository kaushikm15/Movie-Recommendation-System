##################################################
################ IMPORT STATEMENTS ###############
##################################################
import numpy as np
import pandas as pd
import dill
import scipy.stats as stats
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from collections import defaultdict
import missingno as msno
import itertools
from surprise import Reader, Dataset, SVDpp

####Constant Declarations####
#datafile_path = './titanic.csv'
pickle_file_path = './SVD_movies.pkl'
movies_df = pd.read_csv('./movies_metadata.csv', low_memory=False)
links_df = pd.read_csv('./links_small.csv', low_memory=False)
ratings_df = pd.read_csv('./ratings_small.csv', low_memory=False)

# Data Cleaning and EDA
def EDA():
    
    # Changing 0s to NaNs.
    movies_df['revenue'] = movies_df['revenue'].replace(0, np.nan)
    movies_df['runtime'] = movies_df['runtime'].replace(0, np.nan)

    # Function to convert non-numeric values to NaN
    def clean_numeric(x):
        try:
            return float(x)
        except:
            return np.nan

    # Converting the values in the Popularity, vote_count and vote_average column to make sure they have either 
    # numbers or NaNs
    movies_df['popularity'] = movies_df['popularity'].apply(clean_numeric).astype('float')
    movies_df['vote_count'] = movies_df['vote_count'].apply(clean_numeric).astype('float')
    movies_df['vote_average'] = movies_df['vote_average'].apply(clean_numeric).astype('float')

    # Since budget has a few text entries instead of numbers, we will deal with them here
    # and convert the column from Object to numeric type
    movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
    movies_df['budget'] = movies_df['budget'].replace(0, np.nan)

    # Dropping the adult column as it contains limited information to use
    movies_df.drop('adult', axis=1, inplace=True)

    # Adding the poster path obtained by inspecting the image element on TMDb web page
    base_poster_path = 'http://image.tmdb.org/t/p/w185/'
    movies_df['poster_path'] = "<img src='" + base_poster_path + movies_df['poster_path'] + "' style='height:100px;'>"

    # Changing the datatype of the id feature from the movies dataframe
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')

    # Merging ratings, links and movies dataframes
    lr_df = pd.merge(links_df, ratings_df, how='inner', on='movieId')
    lr_df.head()
    lr_df.columns = ['movieId','imdbId','id','userId','rating','timestamp']
    mlr_df = pd.merge(movies_df, lr_df, how='inner', on='id')

    return mlr_df

# Model fitting
def fit_model(mlr_df):
    algo = SVDpp()
    # Object to parse the data
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(mlr_df[['userId', 'id', 'rating']],reader)
    trainset = data.build_full_trainset()
    PREDICTOR = algo.fit(trainset)
    return PREDICTOR

# Serialize
def serialization(PREDICTOR):
    with open(pickle_file_path, 'wb') as f:
        dill.dump(PREDICTOR, f)

# Main Function        
def main():
    try:
        mlr_df = EDA()
        PREDICTOR = fit_model(mlr_df)
        serialization(PREDICTOR)
        print('SVD model is Trained and Serialized')
    except Exception as err:
        print(err.args)
        exit

#Program Entry Function
if __name__ == '__main__':
    main()