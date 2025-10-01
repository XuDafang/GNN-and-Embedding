import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

# Update some engagement counts and add/delete new engagements (change edge weights and add some edges)
# Configuration
DATA_DIR = 'data'
ENGAGEMENT_TYPES = [
    'like', 'save', 'reshare', 'privateShare', 
    'videoView60', 'profileVisit', 'follow'
]

def main():
    """
    Adds new engagement records between existing users to simulate
    new interactions.
    """
    users_path = os.path.join(DATA_DIR, 'users.csv')
    engagements_path = os.path.join(DATA_DIR, 'engagements.csv')

    if not os.path.exists(users_path) or not os.path.exists(engagements_path):
        print("Original 'users.csv' and 'engagements.csv' not found.")
        print("Please run the initial data_generation.py script first.")
        return

    print("Loading original user and engagement data...")
    users_df = pd.read_csv(users_path)
    engagements_df = pd.read_csv(engagements_path)
    length_before = len(engagements_df)
    NUM_NEW_ENGAGEMENTS = int(length_before * 0.1)  # Add 10% more engagements

    # Get lists of existing users and creators
    actor_ids = users_df[users_df['isCreator'] == False]['id'].tolist()
    creator_ids = users_df[users_df['isCreator'] == True]['id'].tolist()
    
    if not actor_ids or not creator_ids:
        print("Not enough users or creators to generate new engagements.")
        return

    print(f"Generating {NUM_NEW_ENGAGEMENTS} new engagement records...")
    new_engagements = []
    for _ in tqdm(range(NUM_NEW_ENGAGEMENTS)):
        actor_id = random.choice(actor_ids)
        actor_tag = users_df[users_df['id'] == actor_id]['tag'].values

        receiver_id = random.choice(creator_ids)
        rereiver_isCreator = True
        if random.random() <= 0.7:
            # 70% chance to engage with users with same tag
            creator_ids_with_same_tag = users_df[(users_df['isCreator'] == True) & (users_df['tag'] == actor_tag[0])]['id'].tolist()
            normal_user_ids_with_same_tag = users_df[(users_df['isCreator'] == False) & (users_df['tag'] == actor_tag[0])]['id'].tolist()
            receiver_id = random.choice(creator_ids_with_same_tag) 
            if random.random() <= 0.3: # 30% chance to engage with normal users
                rereiver_isCreator = False
                receiver_id = random.choice(normal_user_ids_with_same_tag)  
        else:
            creator_ids_with_diff_tag = users_df[(users_df['isCreator'] == True) & (users_df['tag'] != actor_tag[0])]['id'].tolist()
            normal_user_ids_with_diff_tag = users_df[(users_df['isCreator'] == False) & (users_df['tag'] != actor_tag[0])]['id'].tolist()
            receiver_id = random.choice(creator_ids_with_diff_tag) 
            if random.random() <= 0.3: # 30% chance to engage with normal users
                rereiver_isCreator = False
                receiver_id = random.choice(normal_user_ids_with_diff_tag)  

        # Ensure actor and receiver are different, even though the chance that they are the same is low
        while actor_id == receiver_id:
            receiver_id = random.choice(creator_ids)

        num_engagements = random.randint(2, 3)  # Number of different engagement types to add
        engagement_types = random.sample(ENGAGEMENT_TYPES, num_engagements) # Pick 4 random engagement types
        
        new_row = {
            'actor_id': actor_id,
            'receiver_id': receiver_id,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            'like': 0, 'save': 0, 'reshare': 0, 'privateShare': 0,
            'videoView60': 0, 'profileVisit': 0, 'follow': 0
        }
        for engagement_type in engagement_types:
            new_row[engagement_type] = random.choices([1, 2])[0]  # Random counts for selected types
        if rereiver_isCreator:
            for engagement_type in engagement_types:
                new_row[engagement_type] = random.choices([2, 3, 4])[0]  # Creators get higher engagement counts

        new_engagements.append(new_row)

    new_engagements_df = pd.DataFrame(new_engagements)
    
    # Append to the existing engagements
    updated_engagements_df = pd.concat([engagements_df, new_engagements_df], ignore_index=True)
    
    # Aggregate engagements to ensure one row per user pair
    aggregation_funcs = {eng: 'sum' for eng in ENGAGEMENT_TYPES}
    final_engagements_df = updated_engagements_df.groupby(['actor_id', 'receiver_id']).agg(aggregation_funcs).reset_index()
    

    print("Modifying some existing engagement counts...")
    ################################## Update existing engagements by adding or subtracting some counts ##################################
    for idx, row in final_engagements_df.iterrows():
        if random.random() < 0.1:  # 10% chance to modify an existing engagement
            for eng_type in random.sample(ENGAGEMENT_TYPES, 4): # Pick 4 random engagement types
                if row[eng_type] > 0:
                    change = random.choice([-2, 2]) * random.randint(0, 2)  # Randomly add or subtract up to 2
                    new_count = max(0, row[eng_type] + change)  # Ensure count doesn't go negative
                    final_engagements_df.at[idx, eng_type] = new_count

    # Define output paths for this stage
    users_out_path = os.path.join(DATA_DIR, 'users_1.csv')
    engagements_out_path = os.path.join(DATA_DIR, 'engagements_1.csv')
    
    # Save the updated data for the next step in the pipeline
    print(f"Saving updated data")
    users_df.to_csv(users_out_path, index=False) # Users remain the same
    final_engagements_df.to_csv(engagements_out_path, index=False)
    
    tmp = engagements_df.groupby(['actor_id', 'receiver_id']).agg(aggregation_funcs).reset_index()
    print(f"Original engagements: {len(engagements_df)}")
    print(f"Original engagements after grouping: {len(tmp)}")
    print(f"Engagements after adding new ones: {len(final_engagements_df)}")
    print(f"Data for Case 1 saved to '{engagements_out_path}'")
    
    

if __name__ == '__main__':
    main()
    