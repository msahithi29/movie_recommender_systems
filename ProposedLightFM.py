#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM		
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score
from lightfm.data import Dataset
import pandas as pd

from lightfm.cross_validation import random_train_test_split
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import itertools

# movielens dataset with 100k movie ratings from 1k users on 1700 movies
data = fetch_movielens(min_rating=2.5)
print(data)

train_set = data['train']
test_set = data['test']

# create hybrid model, CB+CF
model = LightFM(learning_rate=0.05,loss='warp')

#train model
model.fit(data['train'],epochs=20,num_threads=2)


# Results
 
result=[]
known_values =[]
k = 10
userID_list = [2,10,60]
known_positives =[]
top_movies = []
def lightfm_recommender(model,data,user_ids):
     # of users and items usinf shape
    no_users,no_movies = data['train'].shape
    
#     generate recommendations for each user we input
    for user_id in user_ids:
#         movies already liked by user so far
        known_positives = data['item_labels'][data['train'].tocsr() [user_id].indices]
#         movies our model predicts they will like
        scores = model.predict(user_id, np.arange(no_movies))
#      rank them in order of most liked to least
        top_movies = data['item_labels'][np.argsort(-scores)]
#       print out the results
        print("User %s" % user_id)
#         userID_list.append(user_id)
        print("     Known positives:")
        
        for x in known_positives[:k]:
            print("        %s" % x)
            known_values.append([user_id,x])
            
        print("     Recommended:")
        for x in top_movies[:k]:
            print("        %s" % x)
            result.append([user_id,x])
    return known_values,result
             
known_values, result = lightfm_recommender(model, data, userID_list)

known_values_df = pd.DataFrame(known_values)
known_values_df.rename(columns={0:'User',1:'Known Positives'},inplace=True)
result_df = pd.DataFrame(result)
result_df.rename(columns={0:'User',1:'Top Movies Recommended'},inplace=True)

result_df

known_values_df


# Evaluation
from lightfm.evaluation import precision_at_k
train_p_at_k = precision_at_k(model, data['train'], k=5).mean()
test_p_at_k = precision_at_k(model, data['test'], k=5).mean()
print("PRECISION@K: Train precision: %.4f" % train_p_at_k)
print("PRECISION@K: Test precision: %.4f" % test_p_at_k)

train_auc = auc_score(model, train_set).mean()
test_auc = auc_score(model, test_set).mean()
print('AUC: train %.4f, test %.4f.' % (train_auc, test_auc))

train_recall = recall_at_k(model, train_set).mean()
test_recall = recall_at_k(model, test_set).mean()
print('RECALL@K: train %.4f, test %.4f.' % (train_recall, test_recall))

def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
            
        }


def random_search_auc(train, test, num_samples=10, num_threads=1,k=5):

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)
        score = auc_score(model, test, train_interactions=train, num_threads=num_threads,check_intersections=False).mean()
        hyperparams["num_epochs"] = num_epochs
        yield (score, hyperparams, model)


if __name__ == "__main__":
    

    (score, hyperparams, model) = max(random_search_auc(train_set, test_set, num_threads=2), key=lambda x: x[0])

    print("Best AUC score {} at {}".format(score, hyperparams))

#Precision@K:

def random_search_precision_at_k(train, test, num_samples=5, num_threads=2,k=5):
    
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        p_at_k_score = precision_at_k(model, test_interactions=test, train_interactions=train, num_threads=num_threads,k=k,check_intersections=False,preserve_rows=True).mean()
        
        hyperparams["num_epochs"] = num_epochs

        yield (p_at_k_score, hyperparams, model)


if __name__ == "__main__":
    

    (p_at_k_score, hyperparams, model) = max(random_search_precision_at_k(train_set, test_set, num_threads=2,k=5), key=lambda x: x[0])

    print("Best Precision@k score {} at {}".format(p_at_k_score, hyperparams))

#Recall@K

def random_search_recall_at_k(train, test, num_samples=10, num_threads=1,k=5):
    

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        r_at_k_score = recall_at_k(model, test, train_interactions=train, num_threads=num_threads, k=k).mean()
        
        hyperparams["num_epochs"] = num_epochs

        yield (r_at_k_score, hyperparams, model)


if __name__ == "__main__":
    

    (r_at_k_score, hyperparams, model) = max(random_search_recall_at_k(train_set, test_set, num_threads=2), key=lambda x: x[0])

    print("Best Recall@k score {} at {}".format(r_at_k_score, hyperparams))


# In[ ]:




