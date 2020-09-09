# movie_recommender_systems

<b> What Should I Watch Next: A Movie Recommendation Study </b>

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
