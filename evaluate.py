# evaluate.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def evaluate_model(mode_type):
    """
    Loads trained embeddings and evaluates recommendation quality.
    """
    print("mode_type:", mode_type)
    # Load data
    data_dir = 'data'
    user_df = pd.read_csv(os.path.join(data_dir, 'users_with_indices.csv'))
    embeddings = np.load(os.path.join(data_dir, 'weighted_final_embeddings.npy'))
    if mode_type == "update_edges":
        embeddings = np.load(os.path.join(data_dir, 'updated_edges_embeddings.npy'))
    # Separate users and creators
    normal_users_df = user_df[user_df['isCreator'] == False]
    creators_df = user_df[user_df['isCreator'] == True]

    # Get embeddings for users and creators
    user_indices = normal_users_df.index
    creator_indices = creators_df.index
    user_embeddings = embeddings[user_indices]
    creator_embeddings = embeddings[creator_indices]

    total_matches = 0
    num_users_evaluated = 0

    print("Evaluating recommendations...")
    # For each user, find top 10 closest creators 
    for i, user_idx in enumerate(tqdm(user_indices)):
        user_info = normal_users_df.loc[user_idx]
        user_emb = user_embeddings[i].reshape(1, -1)
        
        # Calculate cosine similarity 
        sims = cosine_similarity(user_emb, creator_embeddings).flatten()
        
        # Get top 10 creator indices
        top_10_indices = np.argsort(sims)[-10:]
        
        # Get the original IDs of recommended creators
        recommended_creator_ids = creators_df.iloc[top_10_indices]['id']
        recommended_creators_info = creators_df[creators_df['id'].isin(recommended_creator_ids)]
        
        # Compute the percentage of similar creators that have the same tags 
        user_tag = user_info['tag']
        matches = (recommended_creators_info['tag'] == user_tag).sum()
        total_matches += matches
        num_users_evaluated += 1

    # Calculate average percentage
    avg_percentage_match = (total_matches / (num_users_evaluated * 10)) * 100
    print(f"\nAverage Top-10 Tag Match Percentage: {avg_percentage_match:.2f}%")

if __name__ == '__main__':
    mode_type = "original"
    mode_type = "update_edges"  
    evaluate_model(mode_type)