# movie_recommender_systems

<b> What Should I Watch Next: A Movie Recommendation Study </b>
Historically, when searching for the best option, one would typically refer to an expert’s advice for guidance. Who is the best doctor for a specific disease? Which restaurant serves the kind of food that I like the most? All of these kinds of problems require the input and advice of someone with knowledge and expertise. 
These kinds of questions and problems fall under a field of data mining called Recommendation Systems and it is the aim of this paper to develop such a system. Specifically, this paper aims to address the issue of recommending movies that are most likely to be enjoyable to watch for a user.
Though the existing traditional collaborative and content based approaches in recommender systems are effective, these methods have certain drawbacks like cold start, scalability and sparsity in collaborative approach and diversity problem in content based. In this project, we evaluated two hybrid models against two baseline approaches – SVD Content based model using surprise and scikit-learn libraries and LightFM model using Lyst's lightFM library which resolves the issue with cold start, diversity problems and provide relevant high quality recommendations.

<b> Requirements: </b> </br>
Python 3.8 </br>
Numpy 1.14.6 </br>
pandas 1.0.1 </br>
matplotlib </br>
surprise 1.1 </br>
sklearn 0.22 </br>
lightfm 1.15 </br>

<b> Installation statements: </b> </br>
pip install numpy </br>
pip install pandas </br>
pip install matplotlib </br>
pip install scikit-surprise </br>
pip install sklearn </br>
pip install lightfm </br>

<b> Dataset: </b> </br>
Movielens and IMDB datasets should be available in input folder. </br>
Check if the input folder exists. </br>
If not present, create a folder 'input' and copy the input files to this location. </br>
Download the dataset from https://grouplens.org/datasets/movielens/latest/, https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset </br>
</br>
input folder files: </br>
IMDb movies.csv </br>
links.csv </br>
movies.csv </br>
ratings.csv </br>

<b> Four models in four different files: </b> </br>
BaselineKNN.py </br>
BaselineSVD++.py </br>
ProposedHybridSVDContent.py </br>
ProposedLightFM.py </br>
</br>
<b> Same files in jupyter notebook which contains graphs </b>
BaselineKNN.py </br>
BaselineSVD++.py </br>
ProposedHybridSVDContent.py </br>
ProposedLightFM.py </br>


<b> After installing all the required libraries and placing the files in the input folder, run the files using below statements: </b>

python BaselineKNN.py </br>
python BaselineSVD++.py </br>
python ProposedHybridSVDContent.py </br>
python ProposedLightFM.py </br>


<b> References: </b>

1. Harper, F. Maxwell, and Joseph A. Konstan. "The movielens datasets: History and context." Acm transactions on interactive intelligent systems (tiis) 5, no. 4 (2015): 1-19.
2. Hug, Nicolas. "Surprise: A Python library for recommender systems." Journal of Open Source Software 5, no. 52 (2020): 2174.
3. Kula, Maciej. "Metadata embeddings for user and item cold-start recommendations." arXiv preprint arXiv:1507.08439 (2015).
4. Rajaraman, Anand, and Jeffrey David Ullman. Mining of massive datasets. Cambridge University Press, 2011.
