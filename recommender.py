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
    ranked_articles : list, list with top ranked articles (by number of interactions)
    articles_sim_matrix : numpy.array, squared matrix with similarity index between each pair of articles
    user_article_matrix : pandas.DataFrame, matrix where each cell denotes whether there was an interaction for a user-article combination
    svd_user_mat : numpy.array, U matrix obtained from FunkSVD algorithm
    svd_article_mat : numpy.array, transposed V matrix obtained from FunkSVD algorithm
    svd_latent_factors_mat : numpy.array, Sigma matrix obtained from FunkSVD algorithm
    '''
    def __init__(self, articles_df, interactions_df):
        self.articles_df = articles_df
        self.interactions_df = interactions_df
        self.ranked_articles_ = None
        self.articles_sim_matrix_ = None
        self.user_article_matrix_ = None
        self.svd_user_mat_ = None
        self.svd_article_mat_ = None
        svd_latent_factors_mat_ = None

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
        print('Building Ranked Articles DF...')
        self.ranked_articles_ = get_top_article_ids(self.interactions_df.shape[0], self.interactions_df)
        print(self.ranked_articles_[:10], len(self.ranked_articles_))
        print('Building User-Article Interactions Matrix...')
        self.user_article_matrix_ = user_item = create_user_item_matrix(self.interactions_df)
        print(self.user_article_matrix_.head(10), self.user_article_matrix_.shape)
        print('Building Articles Similarity Matrix...')
        self.articles_sim_matrix_ = build_articles_similarity_matrix(self.articles_df)
        print(self.articles_sim_matrix_.shape)
        print(self.articles_sim_matrix_)
        print('Building Users and Movies SVD Matrices...')
        self.svd_user_mat_, self.svd_latent_factors_mat_, self.svd_article_mat_ = np.linalg.svd(self.user_article_matrix_)
        print('S shape:',self.svd_latent_factors_mat_.shape, 
              'U shape:',self.svd_user_mat_.shape, 
              'VT shape:',self.svd_article_mat_.shape)
        
        
    def predict_interaction(self, user_id, article_id):
        '''
        Predicts whether there will be interactions between a given user and article
        
        Parameters
        ----------
        self : object
        user_id : int, ID of user for whom rating prediction will be performed
        article_id : int, ID of article for whom rating prediction will be performed
        
        Returns
        -------
        pred : float, interaction prediction for the user-article combination
        '''
        pred = predict_rating(self.svd_user_mat_, self.svd_article_mat_,
                              self.svd_latent_factors_mat_,
                              self.user_article_matrix_, user_id, article_id)
        return pred

    def make_recs(self, entity_id, entity_type):
        '''
        Given a user id or an article id, make recommendations
        
        Parameters
        ----------
        self : object
        entity_id : int, ID of entity for whom movie recommendation will be performed
        entity_type : string, type of entity (user/article) for whom article recommendation will be performed
        
        Returns
        -------
        rec_ids : numpy.array, recommended articles ids
        rec_names : numpy.array, recommended articles titles
        used_svd : boolean, whether SVD algorithm was used in recommendation
        '''
        recs_ids = []
        recs_names = []
        used_svd = False


        if entity_type == 'article':
            #rec_ids = find_similar_movies(_id, movies, movies_sim_mat)[:rec_num]
            used_funksvd = False

        elif entity_type == 'user':
            #Collaborative Filtering
            print('Using Collaborative Filtering...')
            rec_ids, rec_names = user_user_recs_part2(entity_id, self.user_article_matrix_, 
                                                      self.interactions_df, verbose=True)
    #         #Try FunkSVD
    #         #used_funksvd = True
    #         user_row = get_entity_index(ratings_matrix.index, _id, 'User')
    #         if (user_row != None):
    #             print('Using FunkSVD')
    #             user_preds = np.dot(user_mat[user_row,:],movie_mat)
    #             best_n_preds = user_preds.argsort()[-rec_num:][::-1]
    #             rec_ids = movies.iloc[best_n_preds, ].movie_id.values
    #         else:
    #             #Could not find user in train dataset - try Content Based Approach
    #             print('Using ContentBased')
    #             rec_ids = popular_recommendations(_id,rec_num,ranked_movies)
    #             used_funksvd = False

        else:
            print('Invalid ID Type: {}'.format(entity_type))
            print('Valid options: {article,user}')
            return


    #         recs_ids, recs_names, used_svd = make_recommendations(
    #             entity_id, entity_type, self.user_article_matrix_, 
    #             self.ranked_articles_, self.articles_df, 
    #             self.articles_sim_matrix_, self.svd_user_mat_, 
    #             self.svd_article_mat_, self.svd_latent_factors_mat_)

        return(recs_ids, recs_names, used_svd)


if __name__ == '__main__':
    # test different parts to make sure it works
    # Read in the datasets
    interactions = pd.read_csv('data/clean_interactions.csv')
    articles = pd.read_csv('data/clean_articles_all.csv')
    
    rec = Recommender(articles, interactions)
    #rec = Recommender(articles.iloc[:1000,:], interactions.iloc[:5000,:])
    rec.fit()
    
    #Testing Movie recommendation for within the dataset
    print(rec.make_recs(1, 'user'))
    
    #Testing Rating prediction for users-articles within the dataset
#     for i in range(20):
#         print('Rating Prediction {} - Actual Value: {}'.format(
#             rec.predict_rating(interactions.iloc[i,:].user_id,
#                                interactions.iloc[i,:].article_id),
#             interactions.iloc[i,:].interaction))
    
    #Testing Movie recommendation for users within the dataset
#     recs_dict = {}
#     for i in range(20):
#         user_id = interactions.iloc[i,:].user_id
#         recs_dict[user_id] = rec.make_recs(user_id, 'user')
        
    #Testing Movie recommendation for movies within the dataset
#         article_id = articles.iloc[i,:].article_id
#         recs_dict[article_id] = rec.make_recs(article_id, 'article')
#         print(recs_dict[article_id])

    #Testing Movie recommendation for users outside the dataset
    #print(rec.make_recs(40000, 'user'))
    
