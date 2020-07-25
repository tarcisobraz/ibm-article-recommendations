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

def build_all_articles_dataset(articles_df, interactions_df, verbose=False):
    '''
    INPUT
    articles_df - pandas.DataFrame, dataframe containing article_id and doc_description columns
    interactions_df - (pandas dataframe) dataframe with interactions between users and articles
    verbose - (boolean) wether function should log each step
    
    OUTPUT
    all_articles - pandas.DataFrame, a full version of the articles data frame containing articles 
    which figure in both articles_df and interactions_df
    '''
    extra_articles_ids = np.setdiff1d(interactions_df.article_id.unique(),
                                  articles_df.article_id.unique())
    if verbose: print('extra_articles_ids:', extra_articles_ids)
    extra_articles_data = interactions_df[interactions_df.article_id.isin(extra_articles_ids)] \
                            .groupby(['article_id']).title.first().reset_index()
    extra_articles_data = extra_articles_data.rename(index=str, columns={'title':'doc_description'})
    if verbose: print('extra_articles_data:', extra_articles_data.head())
    all_articles = pd.concat([articles_df, extra_articles_data], axis=0, sort=False)
    all_articles.loc[:,'article_id'] = all_articles.loc[:,'article_id'].astype(str)
    if verbose: print('all_articles:', all_articles.head())
    
    return all_articles
    

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
    most_similar_users = get_top_sorted_users(user_id, interactions_df, user_article_mat).neighbor_id.values
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
    
    return recs

# Content-Based Recommendation

def find_similar_articles(article_id, articles_df, article_sim_mat, 
                          num_recs=-1, min_similarity_threshold=0.95, verbose=False):
    '''
    INPUT
    article_id - int, an article_id 
    articles_df - pandas.DataFrame, dataframe containing article_id and doc_description columns
    article_sim_mat - numpy.array, a square matrix with num_articles rows and columns, where
    num_recs - int, number of recommended articles to return
    every cell represents the cosine similarity between the respective (row and column) articles
    min_similarity_threshold - int, minimum similarity value so the articles can be considered similar
    verbose - boolean, whether or not to print steps logs to std_out
    
    OUTPUT
    similar_articles - an array of the most similar articles by id
    '''
    
    try:
        # find the row of the article id
        article_index = int(articles_df[articles_df['article_id'] == article_id].index.values[0])
        if verbose: print('article_index:', article_index)

        # find the most similar article indices using similarity_threshold
        article_sim_row = article_sim_mat[article_index,]
        if verbose: print('article_sim_row:', article_sim_row)
        mask = (article_sim_row > min_similarity_threshold)
        sort_idx = article_sim_row.argsort()[::-1]
        sim_articles_indices = sort_idx[mask[sort_idx]]
        if verbose: print('sim_articles_indices with actual article:', sim_articles_indices)

        # remove the actual article index from the similar articles list
        similar_articles = sim_articles_indices[sim_articles_indices != article_index].astype(float).astype(str)
        if verbose: print('sim_articles_indices without actual article:', similar_articles)
        
        if num_recs != -1:
            similar_articles = similar_articles[:num_recs]
            if verbose: print('filtered ', num_recs, ' sim_articles_indices:', similar_articles)
                
        return similar_articles
    
    except IndexError:
        print('Article ID {} not found in database'.format(article_id))
        return []
    
def build_user_articles_interactions(interactions_df):
    '''
    INPUT:
    interactions_df - pandas.DataFrame, dataframe containing interactions between users and articles
    
    OUTPUT:
    user_articles_interactions - pandas.DataFrame, a dataframe which holds the total 
    number of interactions between a user-article pair
    
    '''
    user_articles_interactions = interactions_df.groupby(['user_id','article_id']) \
                                    .interaction.count() \
                                    .reset_index() \
                                    .sort_values(['user_id','interaction'],ascending=False)
    user_articles_interactions.loc[:,'article_id'] = user_articles_interactions[['article_id']].astype(str)
    
    return user_articles_interactions
    
def make_content_recs2(user_id, articles_df, user_articles_interactions,
                       article_sim_matrix, num_recs=10, verbose=False):
    '''
    INPUT:
    user_id - int, a user id
    articles_df - pandas.DataFrame, dataframe containing article_id and doc_description columns
    user_articles_interactions - pandas.DataFrame, dataframe containing the number of interactions 
    happened between each pair of user_id and article_id
    article_sim_matrix - numpy.array, a square matrix with num_articles rows and columns, where
    every cell represents the cosine similarity between the respective (row and column) articles
    num_recs - int, number of recommendations to be returned for the user
    verbose - boolean, whether or not to print steps logs to std_out
    
    OUTPUT:
    user_recs - list, a list with articles ids recommended for the user
    
    '''
    # Pull only the articles the user has seen
    user_seen_articles = user_articles_interactions[user_articles_interactions['user_id'] == user_id].article_id.values
    if verbose: print('user_seen_articles:', user_seen_articles)
    # Look at each of the user_seen_articles (higher interactions first)
    user_recs = []
    user_recs_names = []
    
    if len(user_seen_articles) == 0:
        if verbose: print('User has seen no articles - recommending top articles...')
        user_recs = get_top_article_ids(num_recs)
    
    else:    
        for seen_article_id in user_seen_articles:
            if verbose: print('seen_article_id:', seen_article_id)
            # pull the articles the user hasn't seen that are most similar
            seen_article_similar_articles = find_similar_articles(seen_article_id, articles_df, article_sim_matrix)
            if verbose: print('seen_article_similar_articles:', seen_article_similar_articles)
            
            # if no similar article could be found, skip
            if len(seen_article_similar_articles) == 0:
                if verbose: print('No similar article could be found. Skipping...')
                continue

            seen_article_similar_recs = np.setdiff1d(seen_article_similar_articles,
                                                     user_seen_articles,
                                                     assume_unique=True)
            if verbose: print('seen_article_similar_recs:', seen_article_similar_recs)
            
            if verbose: print('user_recs before:', user_recs)                
            user_recs = pd.unique(np.concatenate([user_recs,seen_article_similar_recs]))
            if verbose: print('user_recs after:', user_recs)                

            # These will be the recommendations - continue until num_recs 
            # or you have depleted the movie list for the user
            if user_recs.shape[0] > num_recs:
                user_recs = user_recs[:num_recs]
                if verbose: print('filtered num_recs user_recs:', user_recs)
                break

    return user_recs

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    test_idx = user_item_test.index.values
    test_arts = user_item_test.columns.values
    
    return user_item_train, user_item_test, test_idx, test_arts

def svd_predict_evaluate(train_df, test_df, num_latent_features, verbose=False):
    '''
    INPUT:
    train_df - pandas.DataFrame, dataframe with interactions selected for training
    test_df - pandas.DataFrame, dataframe with interactions selected for testing
    num_latent_features - int, number of latent features to take into consideration
    verbose - boolean, whether or not to print steps logs to std_out
    
    OUTPUT:
    acc - float, accuracy of the model on the test set
    '''
    
    if verbose: print('Creating test and train user-item matrices...')
    user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(train_df, test_df)
    
    if verbose: print('Running Standard SVD Algorithm on train user-item matrix...')
    u_train, s_train, vt_train = np.linalg.svd(user_item_train)
    
    if verbose: print('Filtering test set to keep only users and articles which are within train...')
    test_users_in_train_indices = np.intersect1d(test_idx,user_item_train.index.values)
    test_user_articles_in_train = np.in1d(user_item_train.columns.values, user_item_test.columns.values).nonzero()[0]
    filtered_user_item_test = user_item_test.loc[test_users_in_train_indices,:]
    
    # restructure with k latent features
    if verbose: print('Restructuring SVD matrices with',num_latent_features,'k latent features...')
    s_new, u_new, vt_new = np.diag(s_train[:num_latent_features]), u_train[:, :num_latent_features], vt_train[:num_latent_features, :]
    if verbose: print('s_new.shape:',s_new.shape, 'u_new.shape:',u_new.shape, 'vt_new.shape:',vt_new.shape)

    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    if verbose: print('user_item_est:',user_item_est)

    # select subset of estimation matrix which comprises test intersection users/articles
    test_user_item_est = user_item_est[(test_users_in_train_indices-1)[:, None],test_user_articles_in_train]
    if verbose: print('test_user_item_est:',test_user_item_est.shape,test_user_item_est)

    # compute error for each prediction to actual value
    diffs = np.subtract(filtered_user_item_test, test_user_item_est)
    if verbose: print('diffs:',diffs)

    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    if verbose: print('err:',err)

    #compute accuracy
    acc = 1 - err/(filtered_user_item_test.shape[0]*filtered_user_item_test.shape[1])
    if verbose: print('acc:',acc)
    
    return acc

def get_entity_index(entity_indices, entity_id, entity_name):
    '''
    INPUT:
    entity_indices - numpy array, entity indices in entity matrix
    entity_id - int, entity id in train dataset
    entity_name - string, entity name for error message
    
    OUTPUT:
    entitiy_index - the entity index retrieved from entity indices
    '''
    entity_index = None
    try:
        entity_index = np.where(entity_indices == entity_id)[0][0]
    except IndexError:
        print('{} with id {} not found in train dataset'.format(entity_name, entity_id))
    
    return entity_index

def predict_user_item_int(user_matrix, article_matrix, s_matrix, num_latent_features,
                          users_articles_matrix, user_id, article_id, 
                          verbose=False):
    '''
    INPUT:
    user_matrix - numpy.array, user by latent factor matrix
    article_matrix - numpy.array, latent factor by article matrix
    s_matrix - numpy.array, latent factors diagonal matrix
    num_latent_features - int, number of latent features to take into consideration
    users_articles_matrix - pandas.DataFrame, interactions (user vs articles) matrix
    user_id - int, the user_id from the interactions df
    article_id - int, the article_id from the interactions df
    verbose - boolean, whether or not to print steps logs to std_out
    
    OUTPUT:
    pred_val - the predicted interaction (0 or 1) for user_id-article_id according to SVD
    '''
    # Use the training data to create a series of users and articles that matches the ordering in training data
    user_indices = users_articles_matrix.index
    article_indices = users_articles_matrix.columns.astype(str)
    
    # User row and Article Column
    user_row = get_entity_index(user_indices, user_id, 'User')
    if verbose: print('user_row:',user_row)
    article_col = get_entity_index(article_indices, article_id, 'Article')
    if verbose: print('article_col:',article_col)
    
    if (user_row == None) or (article_col == None):
        if verbose: print('User or article could not be found.')
        return

    # Take dot product of user row, matrix s and article column in U, S and V respectively to make prediction
    pred_mat = np.around(np.dot(np.dot(user_matrix[:,:num_latent_features], np.diag(s_matrix[:num_latent_features])),
                            article_matrix[:num_latent_features,:]))
    pred_val = np.abs(pred_mat[user_row,article_col])
    if verbose: print('pred:',pred_val)
   
    return pred_val

def make_svd_recs(user_id, users_articles_matrix, user_matrix, article_matrix, s_matrix, 
                  num_latent_features, rec_num=5, verbose=False):
    '''
    INPUT:
    user_id - either a user or movie id (int)
    users_articles_matrix - dataframe of data as user-movie matrix
    user_matrix - numpy.array, user by latent factor matrix
    article_matrix - numpy.array, latent factor by article matrix
    s_matrix - numpy.array, latent factors diagonal matrix
    num_latent_features - int, number of latent features to take into consideration
    rec_num - number of recommendations to return (int)
    verbose - boolean, whether or not to print steps logs to std_out
    
    
    OUTPUT:
    rec_ids - (array) a list or numpy array of recommended articles by id                  
    '''
    user_indices = users_articles_matrix.index
    article_indices = users_articles_matrix.columns.astype(str)
    
    user_row = get_entity_index(user_indices, user_id, 'User')
    if verbose: print('user_row:', user_row)
        
    if (user_row == None):
        if verbose: print('User could not be found.')
        return
    
    if verbose: print('Computing Prediction Matrix...')
    pred_mat = np.around(np.dot(np.dot(user_matrix[:,:num_latent_features], 
                                       np.diag(s_matrix[:num_latent_features])),
                        article_matrix[:num_latent_features,:]))
    user_preds = pred_mat[user_row,:]
    if verbose: print('user_preds:',user_preds)
    best_n_preds = user_preds.argsort()[-rec_num:][::-1]
    if verbose: print('best_n_preds:',best_n_preds)
    rec_ids = article_indices[best_n_preds]
    if verbose: print('rec_ids:',rec_ids)
            
    return rec_ids