import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

#Import custom NLP Estimators
import nlp_estimators

def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    interactions_per_article = df.groupby(['article_id','title']).user_id.count() \
                                .reset_index(name='num_interactions')
    top_articles = interactions_per_article.sort_values('num_interactions', 
                                                        ascending=False) \
                                            .title.values[:n].tolist()
    
    
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    interactions_per_article = df.groupby(['article_id','title']).user_id.count() \
                                .reset_index(name='num_interactions')
    top_articles = interactions_per_article.sort_values('num_interactions', 
                                                        ascending=False) \
                                            .article_id.values[:n].tolist()
 
    return top_articles # Return the top article ids

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    df['interaction'] = 1
    user_item = df.groupby(['user_id', 'article_id'])['interaction'].min().unstack().fillna(0)
    
    return user_item # return the user_item matrix 

def build_articles_similarity_matrix(articles_df):
    '''
    INPUT
    articles_df - pandas.DataFrame, dataframe containing article_id and doc_description columns 
    
    OUTPUT
    article_sim_matrix - numpy.array, a square matrix with num_articles rows and columns, where
    every cell represents the cosine similarity between the respective (row and column) articles
    '''
    
    articles_tokens = articles_df[['doc_description']].copy()
    articles_tokens.loc[:,'doc_description'] = articles_tokens.loc[:,'doc_description'].astype(str)

    # Tokenizes strings
    lemmatizer = WordNetLemmatizer()
    articles_tokens.loc[:,'tokens_str'] = articles_tokens.doc_description.apply(
        lambda x: nlp_estimators.tokenize_to_str(x, lemmatizer))
    
    w2v_articles_model = nlp_estimators.TfidfEmbeddingTrainVectorizer(num_dims=300)
    w2v_articles_model.fit(articles_tokens.tokens_str.values, None)
    articles_embeddings = w2v_articles_model.transform(articles_tokens.tokens_str.values)
    article_sim_matrix = cosine_similarity(articles_embeddings,articles_embeddings)
    
    return article_sim_matrix

# Collaborative Filtering

def find_similar_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of the user whose id was given to every other user based on the dot product
    Returns an ordered list of similar users
    
    '''
    # compute similarity of each user to the provided user
    users_similarity = np.dot(user_item.iloc[user_id-1,:],user_item.T)

    # sort by similarity
    # create list of just the ids
    most_similar_users = users_similarity.argsort()[::-1] + 1
   
    # remove the own user's id
    most_similar_users = most_similar_users[most_similar_users != user_id]
       
    return most_similar_users # return a list of the users in order from most to least similar

def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    
    unique_articles = df[['article_id','title']].drop_duplicates() \
                        .set_index('article_id')
    article_names = unique_articles.loc[article_ids,:].title.values
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item, interactions_df):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    interactions_df - (pandas dataframe) dataframe with interactions between users and articles
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    user_row = user_item.iloc[user_id - 1,:]
    article_ids = user_row[user_row > 0].keys().astype(str).values
    article_names = get_article_names(article_ids, interactions_df)
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, user_article_mat, interactions_df, m=10, verbose=False):
    '''
    INPUT:
    user_id - (int) a user id
    user_article_mat - (pandas dataframe) matrix where each cell denotes whether there was an interaction for a user-article combination
    interactions_df - (pandas dataframe) dataframe with interactions between users and articles
    m - (int) the number of recommendations you want for the user
    verbose - (boolean) wether function should log each step
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    recs = np.array([])
    user_seen_ids = get_user_articles(user_id, user_article_mat, interactions_df)[0]
    if verbose: print('user_seen_ids: ', user_seen_ids)
    most_similar_users = find_similar_users(user_id, user_article_mat)
    if verbose: print('most_similar_users: ', most_similar_users)
    for similar_user in most_similar_users:
        if verbose: print('similar_user: ', similar_user)
        sim_user_seen_ids = get_user_articles(similar_user, user_article_mat, interactions_df)[0]
        if verbose: print('sim_user_seen_ids: ', sim_user_seen_ids)
        sim_user_recs = np.setdiff1d(sim_user_seen_ids,user_seen_ids,assume_unique=True)
        if verbose: print('sim_user_recs: ', sim_user_recs)
        if verbose: print('recs before: ', recs, 'size:', recs.shape[0])
        recs = pd.unique(np.concatenate([recs,sim_user_recs]))
        if verbose: print('recs after: ', recs, 'size:', recs.shape[0])
        
    
        if (len(recs) > m):
            recs = recs[:m]
            if verbose: print('filtered m recs: ', recs)
            break
    
    return recs # return your recommendations for this user_id    

def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # compute similarity of each user to the provided user
    users_similarity = np.dot(user_item.iloc[user_id-1,:],user_item.T)
    users_interactions = df.groupby('user_id').interaction.sum() \
                            .reset_index() \
                            .rename(index=str, 
                                    columns={'interaction':'num_interactions',
                                             'user_id':'neighbor_id'})

    # create dataframe with neighbor ids, similarity and number of interactions
    # sort by similarity and number of interactions
    # remove the own user's id
    neighbors_df = pd.DataFrame({'neighbor_id' : user_item.index, 
                                 'similarity' : users_similarity}) \
                    .merge(users_interactions, on='neighbor_id') \
                    .sort_values(['similarity','num_interactions'], ascending=False) \
                    .query('neighbor_id != @user_id') 
   
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, user_article_mat, interactions_df, m=10, verbose=False):
    '''
    INPUT:
    user_id - (int) a user id
    user_article_mat - (pandas dataframe) matrix where each cell denotes whether there was an interaction for a user-article combination
    interactions_df - (pandas dataframe) dataframe with interactions between users and articles
    m - (int) the number of recommendations you want for the user
    verbose - (boolean) whether the output should be verbose (print logs)
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    recs = np.array([])
    interactions_per_article = interactions_df.groupby(['article_id','title']).user_id.count() \
                                .reset_index(name='num_interactions') \
                                .assign(article_id = lambda x: x.article_id.astype(str))
    user_seen_ids = get_user_articles(user_id, user_article_mat, interactions_df)[0]
    if verbose: print('user_seen_ids:', user_seen_ids)
    most_similar_users = get_top_sorted_users(user_id, user_article_mat).neighbor_id.values
    if verbose: print('most_similar_users:', most_similar_users)
    for similar_user in most_similar_users:
        if verbose: print('similar_user:', similar_user)
        sim_user_seen_ids = get_user_articles(similar_user, user_article_mat, interactions_df)[0]
        if verbose: print('sim_user_seen_ids:', sim_user_seen_ids)
        sim_user_recs = np.setdiff1d(sim_user_seen_ids,
                                     np.concatenate([user_seen_ids,recs]),assume_unique=True)
        if verbose: print('sim_user_recs', sim_user_recs, sim_user_recs.shape[0])
        ordered_sim_user_recs = interactions_per_article[interactions_per_article['article_id']\
                                                         .isin(sim_user_recs)]\
                                    .sort_values('num_interactions', ascending=False).article_id.values
        if verbose: print('ordered_sim_user_recs', ordered_sim_user_recs, ordered_sim_user_recs.shape[0])
        if verbose: print('recs before', recs, 'size:', recs.shape[0])
        recs = pd.unique(np.concatenate([recs,ordered_sim_user_recs]))
        if verbose: print('recs after', recs, 'size:', recs.shape[0])
    
        if (len(recs) > m):
            recs = recs[:m]
            if verbose: print('filtered m recs', recs)
            break
            
    rec_names = get_article_names(recs)
    
    return recs, rec_names

# Content-Based Filtering

# Matrix Factorization Recommendation (SVD)