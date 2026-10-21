import os
import shutil
import pandas as pd
import kagglehub

def prepare_data():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/semeval2014", exist_ok=True)

    amazon_path = kagglehub.dataset_download("grikomsn/amazon-cell-phones-reviews")
    reviews_file = os.path.join(amazon_path, "20191226-reviews.csv")
    
    df = pd.read_csv(reviews_file)
    df_clean = df.dropna(subset=['body']).copy()
    df_clean = df_clean[df_clean['body'].str.strip().str.len() > 10]

    sample_df = df_clean.sample(n=min(500, len(df_clean)), random_state=42)
    sample_path = os.path.join("data", "sample_amazon_reviews.csv")
    sample_df.to_csv(sample_path, index=False)

    semeval_path = kagglehub.dataset_download("charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis")
    for item in os.listdir(semeval_path):
        s = os.path.join(semeval_path, item)
        d = os.path.join("data", "semeval2014", item)
        if os.path.isfile(s):
            shutil.copy2(s, d)

if __name__ == "__main__":
    prepare_data()
