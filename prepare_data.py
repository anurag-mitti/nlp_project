"""
Data Preparation Script
Extracts a clean sample of 500 Amazon smartphone reviews for batch exploration.
"""

import os
import pandas as pd
import kagglehub

def prepare_data():
    os.makedirs("data", exist_ok=True)
    sample_path = os.path.join("data", "sample_reviews.csv")
    
    if os.path.exists(sample_path):
        print(f"Sample reviews already prepared at {sample_path}")
        return

    print("Loading Amazon Cell Phones Reviews dataset from cache...")
    amazon_path = kagglehub.dataset_download("grikomsn/amazon-cell-phones-reviews")
    reviews_file = os.path.join(amazon_path, "20191226-reviews.csv")
    
    df = pd.read_csv(reviews_file)
    df_clean = df.dropna(subset=['body']).copy()
    df_clean = df_clean[df_clean['body'].str.strip().str.len() > 15]
    
    sample_df = df_clean.sample(n=min(500, len(df_clean)), random_state=42)
    sample_df[['asin', 'rating', 'title', 'body']].to_csv(sample_path, index=False)
    print(f"Saved {len(sample_df)} sample reviews to {sample_path}")

if __name__ == "__main__":
    prepare_data()
