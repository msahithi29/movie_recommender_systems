import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from collections import defaultdict
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#reading files
df_ratings = pd.read_csv('input/ratings.csv')
df_movies = pd.read_csv('input/movies.csv')
df_ratings = df_ratings.drop(columns= 'timestamp')
print(df_movies.head(5))
print(df_ratings.head(5))
#splitting data into train and test sets
train_split, test_split = train_test_split(df_ratings, test_size = 0.25, random_state = 20)
print("Training data size:", train_split.shape)   
print("Test data size:", test_split.shape)
#reader for parsing the ratings file
reader = Reader(rating_scale=(1, 5))
#building the train and test set, loading the data from dataframe
train_build = Dataset.load_from_df(train_split, reader)
test_build = Dataset.load_from_df(test_split, reader)
trainset = train_build.build_full_trainset()
testset = test_build.build_full_trainset().build_testset()
print("Test set size:", len(testset))
#model building
#takes in factors, epochs, learning rate and regularization parameter
model = SVDpp(n_factors=20,n_epochs=5,lr_all=0.09,reg_all=0.5) 
model.fit(trainset) 
#making predictions
predictions = model.test(testset) 
#calculating rmse
accuracy.rmse(predictions, verbose = True)
#Save all the predicted ratings and convert it to a dataframe
all_recommendations_list = defaultdict(list)
all_recommendations_df = pd.DataFrame([])
for uid, iid, true_r, est, _ in predictions:
    all_recommendations_list[uid].append((iid, est))
    all_recommendations_df = all_recommendations_df.append(pd.DataFrame({'user': uid, 'movieId': iid, 'predicted_rating' : est}, index=[0]), ignore_index=True);
print(all_recommendations_df.head(5))
print(all_recommendations_df.shape)
#Merging with movies file to get genre, title information for predictions
all_recommendations_df_details = pd.merge(all_recommendations_df,df_movies, on='movieId', how='inner')
print(all_recommendations_df_details.head(5))
#List of top n recommendations list as per SVD++
def get_top_n_recommendation_list_df(all_recommendations_df_details, n=10):
    top_n_recommendations_df = all_recommendations_df_details.sort_values(['user','predicted_rating'],ascending=[True, False])
    return top_n_recommendations_df
top_n_recommendations_df = get_top_n_recommendation_list_df(all_recommendations_df_details, 10)
print(top_n_recommendations_df.head())
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
            #here
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
#Calculate precision and recall at n
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
for topn in range(2,20):
    precision, recall = get_precision_recall_at_n(predictions,topn,rating_threshold)
    precision_at_n = sum(prec for prec in precision.values()) / len(precision)
    recall_at_n = sum(rec for rec in recall.values()) / len(recall)   
    precision_recall_at_n.append({'topN' : topn, 'Precision' : precision_at_n, 'Recall': recall_at_n})
print("Precision and recall:")
for n in range(3,9):
    print(precision_recall_at_n[n])    
#get user high rated and liked movies
all_movie_df_details = pd.merge(df_ratings,df_movies, on='movieId', how='inner')
all_movie_df_details = all_movie_df_details.sort_values(['userId','rating'],ascending=[True, False])
print("Top 10 highly rated movies by user:")
print(all_movie_df_details.loc[all_movie_df_details['userId'] == 10].head(10)) #user 10 top 10 rated movies
#user 10 top 10 movie recommendations list
print("Top 10 list of recommendations:")
print(top_n_recommendations_df.loc[top_n_recommendations_df['user'] == 10].head(10))

