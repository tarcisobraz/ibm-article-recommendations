## IBM Watson Article Recommendations

### Table of Contents

1. [Installation](#installation)
2. [Motivation](#motivation)
3. [Repository Structure / Files](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code assumes you use Anaconda (Python 3) with the following extra libraries installed: gensim (use the following installation command): `conda install -c anaconda gensim`

## Motivation<a name="motivation"></a>

As part of the activities of Udacity's Data Science Nanodegree I'm enrolled, I created this project, which aims at implementing a set of Recommendation algorithms applied to a real world dataset.

Recommendation Systems is a very hot topic nowadays, as people use the internet to find and consume a variety of products and media, being central to the business of companies like Amazon, Google, Netflix, Spotify and AirBnB.

In this project, I use a dataset comprised of IBM Watson Studio platform's users, articles and their interactions (accesses to articles by users).
With such data, I am able to build several recommenders using different algorithms, such as: Rank-Based, Content-Based, Collaborative Filtering and Matrix Factorization.
Using these models, I can make recommendations of articles to users, as well as find similar articles to a given article.

## Repository Structure / Files <a name="files"></a>

- The `data` folder comprises the dataset's raw data as well as some preprocessed versions of it to faccilitate their use in the recommendation algorithms. 
- The `test` folder contains files test code and files to test some developed functions.

## Results<a name="results"></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [IBM Watson](https://www.ibm.com/watson) for the data. 
Feel free to use the code provided that you give credits / cite this repo, as well as to contribute.
