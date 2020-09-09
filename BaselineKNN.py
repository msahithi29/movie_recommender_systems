from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from collections import defaultdict
import numpy as np
from surprise import KNNBasic

df_ratings = pd.read_csv('input/ratings.csv')
df_movies = pd.read_csv('input/movies.csv')
df_ratings = df_ratings.drop(columns='timestamp')

# loading the data
train_split, test_split = train_test_split(df_ratings, test_size = 0.3, random_state=42)
reader = Reader(rating_scale=(1,5))
train_build = Dataset.load_from_df(train_split, reader)
test_build = Dataset.load_from_df(test_split, reader)
trainset = train_build.build_full_trainset()
testset = test_build.build_full_trainset().build_testset()

model = KNNBasic(k=50,min_k=20)
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions, verbose=True)

metrics=[]
true_pos_array = []
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
            true_pos_array.append(true_r)
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
for x in true_pos_array:
    if x >= rating_threshold:
        x = 1
    else:
        x = 0
    true_bin_array.append(x)
auc_score = roc_auc_score(true_bin_array,est_array,multi_class='raise',average='macro')    
print('AUC Score: ',auc_score)

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
all_precision = 0
all_recall = 0
for topn in range(2,20):
    precision, recall = get_precision_recall_at_n(predictions,topn,rating_threshold)
    precision_at_n = sum(prec for prec in precision.values()) / len(precision)
    recall_at_n = sum(rec for rec in recall.values()) / len(recall)   
    precision_recall_at_n.append({'topN' : topn, 'Precision' : precision_at_n, 'Recall': recall_at_n})
print("Precision and recall:")
print(precision_recall_at_n)

userId = 10
# Display top rated movies
def get_top_n_rated_movies_for(userId, n=10):
    r = df_ratings[df_ratings['userId'] == userId]
    r_sorted = r.sort_values('rating', ascending=False)
    top_rated_movies = r_sorted.head(n)
    movie_info = df_movies[df_movies['movieId'].isin(top_rated_movies['movieId'])]
    return movie_info

top_rated_movies = get_top_n_rated_movies_for(userId)
print("Top 10 highly rated movies by user:")
print(top_rated_movies.head(20))

# Display predictions

def get_top_n_recommended_movies_for(userId, n=10):
    movies = list(filter(lambda p: p[0] == userId, predictions))
    movies_sorted = sorted(movies, key=lambda p: p.est, reverse=True)
    movieIds = [p.iid for p in movies_sorted]
    return df_movies[df_movies['movieId'].isin(movieIds)][:n]

recommended_movies = get_top_n_recommended_movies_for(userId)
print("Top 10 list of recommendations:")
print(recommended_movies.head(20))
