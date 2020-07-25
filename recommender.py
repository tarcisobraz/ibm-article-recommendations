import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

from recommender_functions import *

class Recommender():
    '''
    This class holds methods to fit and use an IBM Watson article recommender for users
    Parameters
    ----------
    articles_df : pandas.DataFrame, dataframe with articles and their characteristics
    interactions_df : pandas.DataFrame, dataframe with interactions between users and articles
    
    Attributes
    ----------
    all_articles : pandas.DataFrame, a full version of the articles data frame containing articles 
    which figure in both articles_df and interactions_df
    ranked_articles : list, list with top ranked articles (by number of interactions)
    articles_sim_matrix : numpy.array, squared matrix with similarity index between each pair of articles
    user_article_matrix : pandas.DataFrame, matrix where each cell denotes whether there was an interaction for a user-article combination
    svd_user_mat : numpy.array, U matrix obtained from FunkSVD algorithm
    svd_article_mat : numpy.array, transposed V matrix obtained from FunkSVD algorithm
    svd_latent_factors_mat : numpy.array, Sigma matrix obtained from FunkSVD algorithm
    user_articles_interactions : pandas.DataFrame, dataframe with number of interactions between users and articles in the interactions_df
    '''
    def __init__(self, articles_df, interactions_df):
        self.articles_df = articles_df
        self.interactions_df = interactions_df
        self.all_articles_ = None
        self.ranked_articles_ = None
        self.articles_sim_matrix_ = None
        self.user_article_matrix_ = None
        self.svd_user_mat_ = None
        self.svd_article_mat_ = None
        svd_latent_factors_mat_ = None
        user_articles_interactions_ = None

    def fit(self):
        '''
        Builds all intermediate datasets to be used for prediction / recommendation
        using movies and reviews data.
            - Builds ranked_articles df
            - Builds articles_sim_matrix
            - Builds user_article_matrix for SVD computation
            - Builds users, articles and sigma matrices using SVD algorithm
        
        Parameters
        ----------
        self : object
        
        Returns
        -------
        self : object
            Returns self.
        '''
        print('articles_df.shape:', self.articles_df.shape)
        print('interactions_df.shape:', self.interactions_df.shape)
        print('Setting article id to string in both input dfs...')
        self.articles_df.loc[:,'article_id'] = self.articles_df.loc[:,'article_id'].astype(str)
        self.interactions_df.loc[:,'article_id'] = self.interactions_df.loc[:,'article_id'].astype(str)
        print('articles_df types:', self.articles_df.dtypes)
        print('interactions_df types:', self.interactions_df.dtypes)
        print('Building Full Articles Dataset...')
        self.all_articles = build_all_articles_dataset(self.articles_df, self.interactions_df)
        print('Building Ranked Articles DF...')
        self.ranked_articles_ = get_top_article_ids(self.interactions_df.shape[0], self.interactions_df)
        print(self.ranked_articles_[:10], len(self.ranked_articles_))
        print('Building User-Article Interactions Matrix...')
        self.user_article_matrix_ = user_item = create_user_item_matrix(self.interactions_df)
        print(self.user_article_matrix_.head(10), self.user_article_matrix_.shape)
        print('Building Articles Similarity Matrix...')
        self.articles_sim_matrix_ = build_articles_similarity_matrix(self.all_articles)
        print(self.articles_sim_matrix_.shape)
        print(self.articles_sim_matrix_)
        print('Building User Articles Interactions DataFrame...')
        self.user_articles_interactions_ = build_user_articles_interactions(self.interactions_df)
        print(self.user_articles_interactions_.head())
        print('Building Users and Movies SVD Matrices...')
        self.svd_user_mat_, self.svd_latent_factors_mat_, self.svd_article_mat_ = np.linalg.svd(self.user_article_matrix_)
        print('S shape:',self.svd_latent_factors_mat_.shape, 
              'U shape:',self.svd_user_mat_.shape, 
              'VT shape:',self.svd_article_mat_.shape)
        
    def predict_interaction(self, user_id, article_id, num_latent_features=10, verbose=False):
        '''
        Predicts whether there will be interactions between a given user and article
        
        Parameters
        ----------
        self : object
        user_id : int, ID of user for whom interaction prediction will be performed
        article_id : int, ID of article for whom interaction prediction will be performed
        
        Returns
        -------
        pred : float, interaction prediction (0 or 1) for the user-article combination
        '''
        pred = predict_user_item_int(self.svd_user_mat_, self.svd_article_mat_,
                              self.svd_latent_factors_mat_, num_latent_features,
                              self.user_article_matrix_, user_id, article_id,
                              verbose)
        return pred
    
    def predict_evaluate(self, train_interactions_df, test_interactions_df, 
                         num_latent_features, verbose=False):
        acc = svd_predict_evaluate(train_interactions_df, test_interactions_df, 
                                   num_latent_features, verbose=verbose)
        return acc
        

    def make_recs(self, entity_id, entity_type, num_recs=10, rec_type=1, verbose=False):
        '''
        Given a user id or an article id, make recommendations
        
        Parameters
        ----------
        self : object
        entity_id : int, ID of entity for whom movie recommendation will be performed
        entity_type : string, type of entity (user/article) for whom article recommendation will be performed
        rec_type : int, type of recommendation algorithm to be used:
                        1 - User-User Collaborative Filtering Recommendation
                        2 - Content-Based Recommendation
                        3 - Matrix Factorization Recommendation
        verbose : boolean, whether the output should be verbose (print logs)
        
        Returns
        -------
        rec_ids : numpy.array, recommended articles ids
        rec_names : numpy.array, recommended articles titles
        '''
        recs_ids = []
        recs_names = []

        if entity_type == 'article':
            if verbose: print('Article Recommendation for article ID:', entity_id)
            if verbose: print('Using Content-Based Recommendation...')
            recs_ids = find_similar_articles(article_id, 
                                             self.all_articles, 
                                             self.articles_sim_matrix_, 
                                             num_recs=num_recs, 
                                             verbose=verbose)

        elif entity_type == 'user':
            if verbose: print('Article Recommendation for user ID:', entity_id)
            if rec_type == 1:
                #Collaborative Filtering
                if verbose: print('Using Collaborative Filtering Recommendation...')
                recs_ids = user_user_recs_part2(entity_id, 
                                                self.user_article_matrix_, 
                                                self.interactions_df, 
                                                m=num_recs, 
                                                verbose=verbose)
            elif rec_type == 2:
                #Content-Based
                if verbose: print('Using Content-Based Recommendation...')
                recs_ids = make_content_recs2(entity_id, 
                                              self.all_articles, 
                                              self.user_articles_interactions_, 
                                              self.articles_sim_matrix_,
                                              num_recs=num_recs,
                                              verbose=verbose)
                
            elif rec_type == 3:
                #Matrix Factorization (SVD)
                if verbose: print('Using Matrix Factorization (SVD) Recommendation...')
                recs_ids = make_svd_recs(entity_id, self.user_article_matrix_, 
                                         self.svd_user_mat_, self.svd_article_mat_,
                                         self.svd_latent_factors_mat_, num_latent_features=10, 
                                         rec_num=5, verbose=verbose)
            
            else:
                print('Invalid Rec Type: {}'.format(rec_type))
                print('Valid options: {1,2,3,4}')
                print('1 - User-User Collaborative Filtering Recommendation \n \
                       2 - Content-Based Recommendation \n \
                       3 - Matrix Factorization Recommendation')
                return

        else:
            print('Invalid ID Type: {}'.format(entity_type))
            print('Valid options: {article,user}')
            return


        #         recs_ids, recs_names, used_svd = make_recommendations(
        #             entity_id, entity_type, self.user_article_matrix_, 
        #             self.ranked_articles_, self.articles_df, 
        #             self.articles_sim_matrix_, self.svd_user_mat_, 
        #             self.svd_article_mat_, self.svd_latent_factors_mat_)
        
        rec_names = get_article_names(recs_ids, self.interactions_df)

        return(recs_ids, recs_names)


if __name__ == '__main__':
    # test different parts to make sure it works
    # Read in the datasets
    interactions = pd.read_csv('data/clean_interactions.csv')
    articles = pd.read_csv('data/clean_articles_all.csv')
    
    rec = Recommender(articles, interactions)
    #rec = Recommender(articles.iloc[:1000,:], interactions.iloc[:5000,:])
    rec.fit()
    
    #Testing Movie recommendation for within the dataset
    #print(rec.make_recs(1, 'user'))
    
    #Testing SVD prediction evaluation function
    df_train = interactions.head(40000).copy()
    df_test = interactions.tail(5993).copy()
    rec.predict_evaluate(df_train, df_test, 10, True)
    
    #Testing SVD prediction for a user-article pair
    #rec.predict_interaction(user_id=8, article_id='1429.0', verbose=True)
    
    #Testing Interaction prediction for users-articles within the interactions dataset
    for i in range(20):
        user_id = interactions.iloc[i,:].user_id
        article_id = interactions.iloc[i,:].article_id
        print('Interaction Prediction between User {} and Article {}: {}'.format(
            user_id,article_id,rec.predict_interaction(user_id,article_id)))
    
    #Testing Collaborative-Filtering Article recommendation for users within the dataset
    recs_dict = {}
    for i in range(5):
        print()
        print()
        print('================')
        user_id = interactions.iloc[i,:].user_id
        print('Making Recommendations for User ID:', user_id)
        recs_dict[user_id] = rec.make_recs(user_id, 'user', verbose=True)
        
    #Testing Content-Based Article recommendation for users within the dataset
    recs_dict = {}
    for i in range(5):
        print()
        print()
        print('================')
        user_id = interactions.iloc[i,:].user_id
        print('Making Recommendations for User ID:', user_id)
        recs_dict[user_id] = rec.make_recs(user_id, 'user', rec_type=2, verbose=True)
        
    #Testing Article recommendation for articles within the dataset
    recs_dict = {}
    for i in range(5):
        print()
        print()
        print('================')
        article_id = interactions.iloc[i,:].article_id
        print('Making Recommendations for Article ID:', article_id)
        recs_dict[article_id] = rec.make_recs(article_id, 'article', verbose=True)

    #Testing SVD Article recommendation for users within the dataset
    recs_dict = {}
    for i in range(5):
        print()
        print()
        print('================')
        user_id = interactions.iloc[i,:].user_id
        print('Making Recommendations for User ID:', user_id)
        recs_dict[user_id] = rec.make_recs(user_id, 'user', rec_type=3, verbose=True)
        

    #Testing Movie recommendation for users outside the dataset
    #print(rec.make_recs(40000, 'user'))
    
