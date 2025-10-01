import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os  # Added import for directory handling

from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 100000
NUM_CREATORS = 600
NUM_GROUPS = 20
USERS_PER_GROUP = NUM_USERS // NUM_GROUPS
CREATORS_PER_TAG = NUM_CREATORS // NUM_GROUPS

# Interest tags
TAGS = ['finance', 'sports', 'technology', 'movie', 'travelling', 
        'politics', 'education', 'adventure', 'games', 'music',
        'food', 'fashion', 'health', 'science', 'history',
        'art', 'nature', 'photography', 'business', 'literature']

# Engagement types with their weights (importance)
ENGAGEMENT_TYPES = {
    'like': 1.0,
    'save': 1.5,
    'reshare': 2.0,
    'privateShare': 1.8,
    'videoView60': 1.2,
    'profileVisit': 1.1,
    'follow': 3.0
}

def generate_users():
    """Generate users and creators"""
    users = []
    user_id = 100000  # Start from 6-digit ID
    
    # Generate normal users
    for group_idx in range(NUM_GROUPS):
        tag = TAGS[group_idx]
        
        for user_in_group in range(USERS_PER_GROUP):
            # Normal users have fewer followers (0-1000)
            follower_count = np.random.poisson(50)  # Most users have few followers
            
            users.append({
                'id': user_id,  # Changed from 'ID' to 'id' for consistency
                'isCreator': False,
                'tag': tag,
                'followerCount': min(follower_count, 1000)  # Cap at 1000
            })
            user_id += 1
    
    # Generate creators
    creator_followers = defaultdict(list)
    
    for tag in TAGS:
        for _ in range(CREATORS_PER_TAG):
            # Creators have more followers - using power law distribution
            base_followers = np.random.pareto(2.0) * 5000 + 1000
            follower_count = min(int(base_followers), 1000000)  # Cap at 1M
            
            users.append({
                'id': user_id, 
                'isCreator': True,
                'tag': tag,
                'followerCount': follower_count
            })
            
            creator_followers[tag].append((user_id, follower_count))
            user_id += 1
    
    return pd.DataFrame(users), creator_followers

def generate_engagements(users_df, creator_followers):
    """Generate engagement data with unique (actor, receiver) pairs"""
    engagements = []
    user_ids = users_df[users_df['isCreator'] == False]['id'].tolist()
    creator_ids = users_df[users_df['isCreator'] == True]['id'].tolist()
    
    # Create mapping for quick lookups
    user_tags = {row['id']: row['tag'] for _, row in users_df.iterrows()}
    user_followers = {row['id']: row['followerCount'] for _, row in users_df.iterrows()}
    
    # Pre-calculate creator engagement probabilities based on followers
    creator_weights = {}
    for creator_id in creator_ids:
        creator_weights[creator_id] = user_followers[creator_id] ** 0.7
    
    total_creator_weight = sum(creator_weights.values())
    creator_probs = {cid: weight/total_creator_weight for cid, weight in creator_weights.items()}
    
    # Track which pairs we've already created to avoid duplicates
    created_pairs = set()
    
    # Generate engagements for each user
    for actor_id in tqdm(user_ids):
        actor_tag = user_tags[actor_id]
        
        # Engage with 10 creators (7 same tag, 3 different)
        creator_pool_same_tag = [cid for cid in creator_ids if user_tags[cid] == actor_tag and (actor_id, cid) not in created_pairs]
        creator_pool_diff_tag = [cid for cid in creator_ids if user_tags[cid] != actor_tag and (actor_id, cid) not in created_pairs]
        
        same_tag_creators = random.choices(
            creator_pool_same_tag, 
            weights=[creator_probs[cid] for cid in creator_pool_same_tag], 
            k=min(7, len(creator_pool_same_tag))
        ) if creator_pool_same_tag else []
        
        diff_tag_creators = random.choices(
            creator_pool_diff_tag,
            weights=[creator_probs[cid] for cid in creator_pool_diff_tag],
            k=min(3, len(creator_pool_diff_tag))
        ) if creator_pool_diff_tag else []
        
        engaged_creators = same_tag_creators + diff_tag_creators
        
        # Engage with 5 normal users (3 same tag, 2 different)
        user_pool_same_tag = [uid for uid in user_ids if uid != actor_id and user_tags[uid] == actor_tag and (actor_id, uid) not in created_pairs]
        user_pool_diff_tag = [uid for uid in user_ids if uid != actor_id and user_tags[uid] != actor_tag and (actor_id, uid) not in created_pairs]
        
        same_tag_users = random.sample(user_pool_same_tag, min(3, len(user_pool_same_tag))) if user_pool_same_tag else []
        diff_tag_user = random.sample(user_pool_diff_tag, min(2, len(user_pool_diff_tag))) if user_pool_diff_tag else []
        
        engaged_users = same_tag_users + diff_tag_user
        
        # Generate engagements for all receivers
        for idx, receiver_id in enumerate(engaged_creators + engaged_users):
            if actor_id == receiver_id or (actor_id, receiver_id) in created_pairs:
                continue
                
            created_pairs.add((actor_id, receiver_id))
            receiver_tag = user_tags[receiver_id]
            same_tag = (actor_tag == receiver_tag)
            
            # Generate engagement counts
            engagement_counts = {}
            total_engagement_weight = 0
            
            for eng_type, base_weight in ENGAGEMENT_TYPES.items():
                if same_tag:
                    count = random.randint(0, 4)
                else:
                    count = random.randint(0, 2)
                if 1 == random.randint(1, 2):  # 50% chance of no engagement
                    count = 0
                if idx < len(engaged_creators):  # Creators get more engagement
                    count = count * random.randint(2, 3)
                engagement_counts[eng_type] = count
                total_engagement_weight += count * base_weight
            
            # Calculate edge weight (importance)
            edge_weight = total_engagement_weight / sum(engagement_counts.values()) if sum(engagement_counts.values()) > 0 else 0
            
            # Only add engagement if there's at least one interaction
            if sum(engagement_counts.values()) > 0:
                engagement_row = {
                    'actor_id': actor_id,
                    'receiver_id': receiver_id,
                    'receiver_isCreator': receiver_id in creator_ids,
                    'weight': edge_weight
                }
                engagement_row.update(engagement_counts)
                engagements.append(engagement_row)
    
    return pd.DataFrame(engagements)

def main():
    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    print("Generating users and creators...")
    users_df, creator_followers = generate_users()
    
    print("Generating engagements...")
    engagements_df = generate_engagements(users_df, creator_followers)
    
    # Save to CSV files in the data folder
    users_path = os.path.join(data_dir, 'users.csv')
    engagements_path = os.path.join(data_dir, 'engagements.csv')
    
    users_df.to_csv(users_path, index=False)
    engagements_df.to_csv(engagements_path, index=False)
    
    print(f"Generated {len(users_df)} users (including {NUM_CREATORS} creators)")
    print(f"Generated {len(engagements_df)} engagement records")
    print(f"Files saved to: {data_dir}/")
    
    # Print some statistics
    print("\n=== Dataset Statistics ===")
    print(f"Normal users: {len(users_df[users_df['isCreator'] == False])}")
    print(f"Creators: {len(users_df[users_df['isCreator'] == True])}")
    print(f"Users per tag: {USERS_PER_GROUP}")
    print(f"Creators per tag: {CREATORS_PER_TAG}")
    
    # Engagement statistics
    total_engagements = engagements_df[list(ENGAGEMENT_TYPES.keys())].sum().sum()
    print(f"Total engagement actions: {total_engagements}")
    print(f"Average engagements per user: {total_engagements / NUM_USERS:.2f}")
    
    # Sparsity check
    engagement_pairs = len(engagements_df)
    possible_pairs = NUM_USERS * (NUM_USERS - 1)
    sparsity = 1 - (engagement_pairs / possible_pairs)
    print(f"Graph sparsity: {sparsity:.6f}")

if __name__ == "__main__":
    main()