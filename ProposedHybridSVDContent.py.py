import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import accuracy
from collections import defaultdict
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

#Content Based model
#Reading datasets
movies = pd.read_csv("input/movies.csv")
print(movies.head(5))
links = pd.read_csv("input/links.csv")
print(links.head(5))
# converting the imdbId field to be joinable later
links['imdbId2'] = links['imdbId'].map(str).apply(lambda s: "tt0" + s if len(s) == 6 else "tt" + s)
print(links.head(5))
imdb_movies = pd.read_csv("input/IMDb movies.csv")
print(imdb_movies.head(5))
# joining data sets to obtain contextual information on movies (ie. director, actor, production_company)
movies_links = pd.merge(movies, links, how='inner', on='movieId')
print(movies_links.head(5))
movies_complete = pd.merge(movies_links,imdb_movies,how='inner',left_on='imdbId2',right_on='imdb_title_id')

def generate_similarity_matrix():
    mc = movies_complete
    # filling in missing data
    mc['genre'].fillna("No genre")
    mc['language'].fillna("language")
    mc['director'].fillna("director")
    mc['writer'].fillna("writer")
    mc['production_company'].fillna("production_company")
    mc['actors'].fillna("actors")
    movies_complete["tokens"] = mc['genre'] + "," + mc['director'] + mc['writer'] + "," + mc['actors'] + "," + mc['production_company']    
    v = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\s]+')
    tfidf_movies_context_matrix = v.fit_transform(movies_complete['tokens'].values.astype('U'))
    cos_sim_matrix = linear_kernel(tfidf_movies_context_matrix, tfidf_movies_context_matrix)
    return cos_sim_matrix
cos_sim_matrix = generate_similarity_matrix()
ratings = pd.read_csv("input/ratings.csv")
def get_movies_watched_by(userID):
    user_filter = ratings['userId'] == userID
    movies_watched = ratings[user_filter]
    return movies_watched
def get_movie_recommendations_for(userID):
    # get the list of movies that this user has watched
    movies_watched = get_movies_watched_by(userID)
    df_movies_watched = pd.DataFrame()
    for index, row in movies_watched.iterrows():
        i = movies_complete[movies_complete['movieId'] == row['movieId']].index
        df_movies_watched = df_movies_watched.append(movies_complete.loc[i])

    # get similar items based on those movies
    similar_movies = []
    for index, row in df_movies_watched.iterrows():
        # generate top n similar items and add to the similar_movies list
        sim_movies = list(enumerate(cos_sim_matrix[index]))
        sim_movies_sorted = sorted(sim_movies, key=lambda movieid_score_tuple: movieid_score_tuple[1], reverse=True)
        similar_movies = similar_movies + sim_movies_sorted[1:11]   
    # order the list from highest similarity to lowest similarity
    # recommend the top 10 movies
    similar_movies = sorted(similar_movies, key=lambda movieid_score_tuple: movieid_score_tuple[1], reverse=True)
    top_10_recommendations = similar_movies  
    # convert the list of movie indexes to movie id
    results_2 = list()
    for movie_score in top_10_recommendations:
        movie_index = movie_score[0]
        movie_score = movie_score[1]
        movieId = movies_complete.iloc[movie_index]['movieId']
        movieTitle = movies_complete.iloc[movie_index]['title_x']
        
        genre = movies_complete.iloc[movie_index]['genre']
        language = movies_complete.iloc[movie_index]['language']
        director = movies_complete.iloc[movie_index]['director']
        writer = movies_complete.iloc[movie_index]['writer']
        production_company = movies_complete.iloc[movie_index]['production_company']
        actors = movies_complete.iloc[movie_index]['actors']   
        t = (userID, movie_index, movie_score, movieId, movieTitle, genre, language, director, writer, production_company, actors)
        results_2.append(t)
    df_results_2 = pd.DataFrame(results_2, columns=['userID', 'movie_index', 'score', 'movieId', 'title', 'genre', 'language', 'director', 'writer', 'production_company', 'actors']) 
    # need to remove duplicates by only keeping the movie with the highest score
    df_results_2 = df_results_2[df_results_2.groupby(['movieId'],sort=False)['score'].transform(max) == df_results_2['score']]  
    return df_results_2
results = get_movie_recommendations_for(60)
print(results.head(20))

#SVD Model
#reading files
df_ratings = pd.read_csv('input/ratings.csv')
df_movies = pd.read_csv('input/movies.csv')
df_ratings = df_ratings.drop(columns= 'timestamp')
print(df_movies.head(5))
print(df_ratings.head(5))

#splitting data into train and test sets
train_split, test_split = train_test_split(df_ratings, test_size = 0.25, random_state=20)
print("Training data size:", train_split.shape)   
print("Test data size:", test_split.shape) 
#reader to parse the ratings 
reader = Reader(rating_scale=(1, 5))
#Train and test set
train_build = Dataset.load_from_df(train_split, reader)
test_build = Dataset.load_from_df(test_split, reader)
trainset = train_build.build_full_trainset()
testset = test_build.build_full_trainset().build_testset()
print("Test set size:", len(testset))

#Gridsearch to select best parameters
number_of_factors_list = [10,100] 
number_of_epochs_list = [50] 
learning_rate_list = [0.01,0.001] 
regularization_parameter_list = [0.1]
#Ran with the below parameters to find best values. These are commented and gridsearch is performed with less values because it took 3-4 hours to run. 
#number_of_factors_list = [10,20,25,30,35,40,50,100] 
#number_of_epochs_list = [10,20,30,40,50] 
#learning_rate_list = [0.9,0.09,0.009,0.1,0.01,0.001] 
#regularization_parameter_list = [0.9,0.09,0.009,0.1,0.01,0.001] 
hyper_parameters_set = { 'n_factors': number_of_factors_list, 'n_epochs': number_of_epochs_list, 'lr_all': learning_rate_list,'reg_all': regularization_parameter_list} 
trained_model = SVD 
best_model_selection = GridSearchCV(trained_model,hyper_parameters_set,measures=['rmse'], cv=4) 
best_model_selection.fit(train_build) 
print("Best hyperparameters: ",best_model_selection.best_params['rmse'], "to achieve minimum RMSE: " ,best_model_selection.best_score['rmse'])

#Factors vs RMSE
validationset = trainset.build_testset()
training_rmse = []
testing_rmse =[]
number_of_factors_list = [10,20,25,30,35,40,50,100]
for factor in number_of_factors_list:
    model = SVD(n_factors=factor,n_epochs=50,lr_all=0.01,reg_all=0.1)
    model.fit(trainset)
    training_predictions = model.test(validationset)
    training_rmse.append(accuracy.rmse(training_predictions))    
    test_predictions = model.test(testset)
    testing_rmse.append(accuracy.rmse(test_predictions))
plt.figure(0)
plt.plot(number_of_factors_list,testing_rmse, 'b+--', markersize=12, markeredgecolor='r',label='testing')
plt.plot(number_of_factors_list,training_rmse, 'g+--', markersize=12, markeredgecolor='r',label='training')
plt.xlabel('No of factors')
plt.ylabel('RMSE')
plt.title("RMSE vs Factors")
plt.legend()

#Epochs vs RMSE
training_rmse = []
testing_rmse =[]
number_of_epochs_list = [10,20,30,40,50]
for epoch in number_of_epochs_list:
    model = SVD(n_factors=100,n_epochs=epoch,lr_all=0.01,reg_all=0.1)
    model.fit(trainset)
    training_predictions = model.test(validationset)
    training_rmse.append(accuracy.rmse(training_predictions))    
    test_predictions = model.test(testset)
    testing_rmse.append(accuracy.rmse(test_predictions))
plt.figure(1)
plt.plot(number_of_epochs_list,testing_rmse, 'b+--', markersize=12, markeredgecolor='r',label='testing')
plt.plot(number_of_epochs_list,training_rmse, 'g+--', markersize=12, markeredgecolor='r',label='training')
plt.xlabel('No of epochs')
plt.ylabel('RMSE')
plt.title("RMSE vs Epochs")
plt.legend()

#Effect of learning rate with respect to RMSE
training_rmse = []
testing_rmse =[]
learning_rate_list = [0,0.001,0.009,0.01,0.09,0.1] 
for lr in learning_rate_list:
    model = SVD(n_factors=100,n_epochs=50,lr_all=lr,reg_all=0.1)
    model.fit(trainset)
    training_predictions = model.test(validationset)
    training_rmse.append(accuracy.rmse(training_predictions))    
    test_predictions = model.test(testset)
    testing_rmse.append(accuracy.rmse(test_predictions))
plt.figure(2)
plt.plot(learning_rate_list,testing_rmse, 'b+--', markersize=12, markeredgecolor='r',label='testing')
plt.plot(learning_rate_list,training_rmse, 'g+--', markersize=12, markeredgecolor='r',label='training')
plt.xlabel('Learning rate')
plt.ylabel('RMSE')
plt.title("RMSE vs Learning rate")
plt.legend()

#Regularization parameter and RMSE
training_rmse = []
testing_rmse =[]
regularization_parameter_list = [0,0.001,0.009,0.01,0.09,0.1] 
for reg in regularization_parameter_list:
    model = SVD(n_factors=100,n_epochs=50,lr_all=0.01,reg_all=reg)
    model.fit(trainset)
    training_predictions = model.test(validationset)
    training_rmse.append(accuracy.rmse(training_predictions))    
    test_predictions = model.test(testset)
    testing_rmse.append(accuracy.rmse(test_predictions))
plt.figure(3)
plt.plot(regularization_parameter_list,testing_rmse, 'b+--', markersize=12, markeredgecolor='r',label='testing')
plt.plot(regularization_parameter_list,training_rmse, 'g+--', markersize=12, markeredgecolor='r',label='training')
plt.xlabel('Regularization parameter')
plt.ylabel('RMSE')
plt.title("RMSE vs Regularization")
plt.legend()

#Building model using the best parameters from gridsearch
model = SVD(n_factors=100,n_epochs=50,lr_all=0.01,reg_all=0.1) 
model.fit(trainset) 
predictions = model.test(testset) 
accuracy.rmse(predictions, verbose = True)

#Save all the predicted ratings and convert it to a dataframe
all_recommendations_list = defaultdict(list)
all_recommendations_df = pd.DataFrame([])
for uid, iid, true_r, est, _ in predictions:
    all_recommendations_list[uid].append((iid, est))
    all_recommendations_df = all_recommendations_df.append(pd.DataFrame({'user': uid, 'movieId': iid, 'predicted_rating' : est}, index=[0]), ignore_index=True);
print(all_recommendations_df.head(5))
print(all_recommendations_df.shape)

#Append movie info to the predictions
all_recommendations_df_details = pd.merge(all_recommendations_df,df_movies, on='movieId', how='inner')
print(all_recommendations_df_details)
#top n recommendations list
def get_top_n_recommendation_list_df(all_recommendations_df_details, n=10):
    top_n_recommendations_df = all_recommendations_df_details.sort_values(['user','predicted_rating'],ascending=[True, False])
    return top_n_recommendations_df
top_n_recommendations_df = get_top_n_recommendation_list_df(all_recommendations_df_details, n=10)
print(top_n_recommendations_df.head())

#Hybrid model
def hybrid_model(userID):
    content_recommendations_list = get_movie_recommendations_for(userID) #list of movies for that user
    content_recommendations_list= content_recommendations_list[['userID','movieId', 'title', 'genre']]
    for key, columns in content_recommendations_list.iterrows():
        #key is the index of the dataframe, columns are movieid, title and genre
        predict = model.predict(userID, columns["movieId"]) #predicting the rating based on svd model
        content_recommendations_list.loc[key, "predicted rating"] = predict.est #adding a column svd rating and adding prediction value
    return content_recommendations_list.sort_values("predicted rating", ascending=False).iloc[0:11] # return only first 10 movies based on ratings

#calculate evaluation metrics
metrics=[]
true_positives_array = []
est_array = []
for rating_threshold in np.arange(0,5.5,0.5):
    truePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    falsePositives = 0
    accuracy =0
    precision =0
    recall =0
    f1_score = 0
    for uid,_, true_r, est, _ in predictions:
        if(true_r >= rating_threshold and est >= rating_threshold):
            truePositives = truePositives + 1
            true_positives_array.append(true_r)
            est_array.append(est)
        elif(true_r >= rating_threshold and est <= rating_threshold):
            falseNegatives = falseNegatives + 1
        elif(true_r <= rating_threshold and est >= rating_threshold):
            falsePositives = falsePositives + 1
        elif(true_r <= rating_threshold and est <= rating_threshold):
            trueNegatives = trueNegatives + 1
        if(truePositives > 0):
            accuracy = (truePositives + trueNegatives ) / (truePositives + trueNegatives + falsePositives + falseNegatives) 
            precision = truePositives / (truePositives + falsePositives)
            recall = truePositives / (truePositives + falseNegatives)
            f1_score = 2 * (precision * recall) / (precision + recall)           
    metrics.append([rating_threshold,truePositives,trueNegatives,falsePositives,falseNegatives,accuracy,precision,recall,f1_score])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.rename(columns={0:'rating_threshold', 1:'truePositives', 2: 'trueNegatives', 3: 'falsePositives', 4:'falseNegatives', 5: 'Accuracy', 6: 'Precision', 7:'Recall', 8:'F1 Score'},inplace=True)
true_bin_array =[]
for x in true_positives_array:
    if x >= rating_threshold:
        x = 1
    else:
        x = 0
    true_bin_array.append(x)
auc_score = roc_auc_score(true_bin_array,est_array,multi_class='raise',average='macro')    
print('AUC Score: ',auc_score)

#calculate precision @ k and recall @ k
def get_precision_recall_at_n(predictions,topn,rating_threshold):
    all_actual_predicted_list = defaultdict(list)
    precision = dict()
    recall= dict()
    no_of_relevant_items = 0
    no_of_recommended_items_at_top_n = 0
    no_of_relevant_recommended_items_at_top_n = 0
    for uid, iid, true_r, est, _ in predictions:
        all_actual_predicted_list[uid].append((est, true_r))
    for uid, user_ratings in all_actual_predicted_list.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        no_of_relevant_items = sum((true_r >= rating_threshold) for (_, true_r) in user_ratings)
        no_of_recommended_items_at_top_n = sum((est >= rating_threshold) for (est, _) in user_ratings[:topn])
        no_of_relevant_recommended_items_at_top_n = sum(((true_r >= rating_threshold) and (est >= rating_threshold)) for (est, true_r) in user_ratings[:topn])
        precision[uid] = no_of_relevant_recommended_items_at_top_n / no_of_recommended_items_at_top_n if no_of_recommended_items_at_top_n != 0 else 1
        recall[uid] = no_of_relevant_recommended_items_at_top_n / no_of_relevant_items if no_of_relevant_items != 0 else 1    
    return precision, recall
rating_threshold=3
precision_recall_at_n = []
print("Precision and recall:")
for topn in range(2,20):
    precision, recall = get_precision_recall_at_n(predictions,topn,rating_threshold)
    precision_at_n = sum(prec for prec in precision.values()) / len(precision)
    recall_at_n = sum(rec for rec in recall.values()) / len(recall)   
    precision_recall_at_n.append({'topN' : topn, 'Precision' : precision_at_n, 'Recall': recall_at_n})
for n in range(3,9):
    print(precision_recall_at_n[n])

#get user liked and high rated movies
all_movie_df_details = pd.merge(df_ratings,df_movies, on='movieId', how='inner')
all_movie_df_details = all_movie_df_details.sort_values(['userId','rating'],ascending=[True, False])
print("Top 10 highly rated movies by user:")
print(all_movie_df_details.loc[all_movie_df_details['userId'] == 60].head(10))
#output of hybrid model which shows recommendations from user 60
print("Top 10 list of recommendations:")
print(hybrid_model(60))





