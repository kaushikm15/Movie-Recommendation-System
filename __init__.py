import flask
import dill 
import numpy as np
import pandas as pd

# Libraries to import
import numpy as np
import scipy.stats as stats
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from collections import defaultdict
import missingno as msno
from IPython.display import Image, HTML
from scipy.spatial.distance import correlation
import itertools
from surprise import Reader, Dataset, SVD, SVDpp, SlopeOne, accuracy, NormalPredictor, NMF, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
sns.set(style = 'white', color_codes=True)


import warnings; warnings.simplefilter('ignore')

app = flask.Flask(__name__)

# with open('SVD_movies.pkl', 'rb') as f:
#     PREDICTOR = dill.load(f)
##################################
##################################

####Constant Declarations####
pickle_file_path = './SVD_movies.pkl'
movies_df = pd.read_csv('./movies_init.csv', low_memory=False)
mlr_df = pd.read_csv('./mlr_init.csv', low_memory=False)

@app.route('/', methods=['POST', 'GET'])
def page():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       with open('SVD_movies.pkl', 'rb') as f:
           PREDICTOR = dill.load(f)

       inputs = flask.request.form

       name = inputs['name']

       name = int(name)

       #item = pd.DataFrame([[pclass, sex, age, fare, sibsp]], columns=['pclass', 'sex', 'age', 'fare', 'sibsp'])
       print ('User name is: ', name)

       # Converting the movie ids to numeric type to use them in printing the top n recommendations below
       movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')

       movie_favs = []
       # Top N favorite movies of a user
       def favoriteMovies(activeUser, N):
           topMovies = mlr_df[mlr_df.userId==activeUser].sort_values(['rating'], ascending=[0])[:N]
           return list(topMovies.title)

       # Checking the top 3 movies of user with userid=5 to check if the recommendations are in line with their
       # favourite movies
       #print(favoriteMovies(name,3)) 
       movie_favs = favoriteMovies(name,3)


    
       # Get a list of all movie ids
       ids = mlr_df['id'].unique()
       # Get a list of ids that user id has rated
       ids3 = mlr_df.loc[mlr_df['userId'] == name, 'id']
       # Remove the ids that user id has rated from the lis of all movie ids
       ids_to_pred = np.setdiff1d(ids, ids3)

       # Getting the predictions for the movies that the user has not rated
       testset = [[name, id, 4.] for id in ids_to_pred]
       predictions = PREDICTOR.test(testset)
       print(predictions[0:3])

       # Converting the predictions list into a dataframe and extracting the 
       # top recommendations
       predictions_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
       predictions_df = predictions_df.sort_values('est', ascending=False)
       top_recos = predictions_df['iid'][0:3]
       print ('Top recos are:', top_recos)
       

    
       movie_recos = []
       # Printing the top n recommendations for a user with user id u
       for id in top_recos:
           movie_recos.append(movies_df[movies_df['id'] == id]['title'].values[0])
           
       
       #top_reco1 = top_recos[0]
       #top_reco2 = top_recos[1]
       #top_reco3 = top_recos[2]

       #score = PREDICTOR.predict_proba(name)
       #results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       return flask.render_template('dataentrypage.html', top_fav1 = movie_favs[0], top_fav2 = movie_favs[1], top_fav3 = movie_favs[2], top_reco1=movie_recos[0], top_reco2=movie_recos[1], top_reco3=movie_recos[2])
    
    else:
       return flask.render_template('dataentrypage.html', top_fav1 = "Waiting for user input", top_fav2 = "Waiting for user input", top_fav3 = "Waiting for user input", top_reco1="Waiting for user input", top_reco2="Waiting for user input", top_reco3="Waiting for user input")
    
    return flask.render_template('dataentrypage.html', top_fav1 = "Waiting for user input", top_fav2 = "Waiting for user input", top_fav3 = "Waiting for user input", top_reco1="Waiting for user input", top_reco2="Waiting for user input", top_reco3="Waiting for user input")

##################################
if __name__ == '__main__':
    app.run(debug=True)