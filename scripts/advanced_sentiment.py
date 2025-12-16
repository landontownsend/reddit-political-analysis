import sqlite3
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


tqdm.pandas()

def find_database(filename='reddit_data.db'):
    """
    Searches for the database file in common relative locations.
    Returns the full path if found, or None if not found.
    """
    #Get the folder where this script is currently running
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    #List of possible paths to check
    possible_paths = [
        os.path.join(current_dir, filename),                                      
        os.path.join(current_dir, 'reddit-political-analysis', filename),         
        os.path.join(os.path.dirname(current_dir), filename),                     
        os.path.join(os.path.dirname(current_dir), 'reddit-political-analysis', filename) 
    ]

    print(f"Searching for '{filename}'...")
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def run_advanced_sentiment():
    print("--- Starting Advanced Sentiment Analysis (RoBERTa) ---")

    # --- 1. SMART PATH SETUP ---
    db_path = find_database('reddit_data.db')
    
    if db_path is None:
        print("\nERROR: Could not find 'reddit_data.db'.")
        print("Please ensure the database file is in the same folder (or project folder) as this script.")
        return
    else:
        print(f"Success! Found database at: {db_path}")

    
    project_root = os.path.dirname(db_path)
    output_dir = os.path.join(project_root, 'output')

    
    if not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir)

    # --- 2. LOAD DATA ---
    print("Connecting to database...")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT title FROM posts", conn)
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        return
    
    print(f"Loaded {len(df)} headlines.")

    # --- 3. LOAD PRE-TRAINED MODEL ---
    print("Loading pre-trained RoBERTa model (using CPU)...")
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    
    
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1)

    # --- 4. DEFINE ANALYSIS FUNCTION ---
    def get_sentiment(text):
        try:
            
            result = sentiment_task(text[:512])[0]
            label = result['label']
            
            
            
            if label == 'LABEL_0':
                return 'Negative'
            elif label == 'LABEL_1':
                return 'Neutral'
            elif label == 'LABEL_2':
                return 'Positive'
            else:
                return 'Neutral'
        except Exception:
            return 'Neutral'

    # --- 5. RUN ANALYSIS ---
    print("Analyzing sentiment (this might take a minute)...")
    df['sentiment_label'] = df['title'].progress_apply(get_sentiment)

    # --- 6. SAVE RESULTS & VISUALIZE ---
    
    csv_path = os.path.join(output_dir, 'roberta_sentiment_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")

    
    print("Generating chart...")
    plt.figure(figsize=(10, 6))
    
    sns.countplot(x='sentiment_label', data=df, order=['Negative', 'Neutral', 'Positive'], palette='viridis')
    plt.title('Sentiment Distribution of Reddit Political Headlines (RoBERTa)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    img_path = os.path.join(output_dir, 'roberta_sentiment_chart.png')
    plt.savefig(img_path)
    print(f"Chart saved to: {img_path}")
    print("Analysis Complete!")

if __name__ == "__main__":
    run_advanced_sentiment()