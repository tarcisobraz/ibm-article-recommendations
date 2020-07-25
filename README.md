## IBM Watson Article Recommendations

### Table of Contents

1. [Installation](#installation)
2. [Motivation](#motivation)
3. [Repository Structure / Files](#files)
4. [Project Steps](#steps)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code assumes you use Anaconda (Python 3) with the following extra libraries installed: gensim (use the following installation command): `conda install -c anaconda gensim`

## Motivation<a name="motivation"></a>

As part of the activities of Udacity's Data Science Nanodegree I'm enrolled, I created this project, which aims at implementing a set of Recommendation algorithms applied to a real world dataset.

Recommendation Systems is a very hot topic nowadays, as people use the internet to find and consume a variety of products and media, being central to the business of companies like Amazon, Google, Netflix, Spotify and AirBnB.

In this project, I use a dataset comprised of IBM Watson Studio platform's users, articles and their interactions (accesses to articles by users).
With such data, I am able to build several recommenders using different algorithms, such as: Rank-Based, Content-Based, Collaborative Filtering and Matrix Factorization.
Using these models, I can make recommendations of articles to users, as well as find similar articles to a given article.

## Repository Structure / Files <a name="files"></a>

- The `data` folder comprises the dataset's raw data as well as some preprocessed versions of it to faccilitate their use in the recommendation algorithms. It also contains pickles with recommendations made for all users and articles using the different implemented algorithms.
- The `test` folder contains test code and files used to test some developed functions.

On the root folder, it is possible to find a few source files:

- The Recommendations_with_IBM notebook code, which uses code and plots to answer many questions posed by the project template.
- The Recommendations_with_IBM notebook HTML file, which is just an HTML version of the notebook.
- The nlp_estimators.py file, which defines custom NLP scikit-learn estimators and functions used in the different feature sets developed for another project of the same nanodegree. The code in this file was used to extract features from the articles' text in order to compare them to one another, creating a similarity metric.
- The recommender.py file, which defines a Recommender class, encapsulating most of the functions developed in the notebook, faccilitating the (re)use of the implemented Recommendation algorithms.
- The recommender_functions.py file, which defines the functions used by the Recommender class to perform its higher-level operations.

## Project Steps <a name="steps"></a>

The project was divided into sections as follows:

### I. Exploratory Data Analysis

Before making recommendations, some data exploration was performed, answering a couple of questions related to the data, as well as to get acquainted to it.

### II. Rank Based Recommendations

Then, as a first step into recommendations, I've implemented a simple rank-based recommender, which ranks articles based on their total number of interactions amongst the users, and recommends the highest ranked articles. It might be useful for new users (with no previous interactions recorded in the platform).

### III. User-User Based Collaborative Filtering

Going a step further, I've implemented a Collaborative Filtering approach, which compares a given user `u` to the other users in the dataset using their interactions with articles as features. Then, I just recommend to user `u` the articles its most similar users have interacted with, as long as `u` has not seen these articles yet.

### IV. Content Based Recommendations (EXTRA - NOT REQUIRED)

As an extra, I've built a Content-Based Recommender using the articles' textual information present in the dataset. Using an NLP toolkit I've developed in a previous project, I've generated TF-IDF-aggregated word embeddings for each article (using its title/description as input text) and then used cosine similarity to compare between articles. Thereby, I can recommend articles to a user based on the articles which are more similar to the ones he has already interacted with.

### V. Matrix Factorization

Finally, I've implemented a Matrix Factorization Recommendation algorithm, which is able to predict for a given user-article pair whether they will interact or not. I've used Numpy's default SVD implementation to generate the U, S and VT matrices, as it was possible to build the user-article matrix with no null values. This approach also allows for recommendation, as one can easily predict the interaction of a user will all articles in the database and recommend the ones he is more likely to interact with.

## Results<a name="results"></a>

I've used the Recommender class to make recommendations for all users and articles using all suitable implemented algorithms. The sets of recommended articles for each user-method combination is saved in pickle files in the `data` folder.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [IBM Watson](https://www.ibm.com/watson) for the data. 
Feel free to use the code provided that you give credits / cite this repo, as well as to contribute.
