


import pandas as pd
import numpy as np
import re
import sqlite3
from sqlalchemy import create_engine
import os


def load_kaggle_data_to_db():
    """Load into SQLite database"""
    print("Starting data loading...")
    
  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if 'scripts' in script_dir else script_dir
 
    os.chdir(project_root)
    print("Current directory:", os.getcwd())
    
   
    csv_path = os.path.join(project_root, 'data', 'politics.csv')
    
   
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print("Please place politics.csv in the data/ folder")
        return
    
    politics_df = pd.read_csv(csv_path)
    print(f"Loaded {len(politics_df)} rows")
    

    db_path = os.path.join(project_root, 'reddit_data.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


politics_df.to_sql('posts', conn, if_exists='replace', index=False)

print("Database created successfully!")


cursor.execute("SELECT COUNT(*) FROM posts")
count = cursor.fetchone()[0]
print(f" Loaded {count} posts into database")





print("Columns in your dataset:")
print(politics_df.columns.tolist())
print(f"\nFirst few rows:")
print(politics_df.head())


politics_df = politics_df.rename(columns={
    'comms_num': 'num_comments',
    'created': 'created_utc',
    'body': 'selftext'
})


politics_df['subreddit'] = 'politics'  
politics_df['permalink'] = politics_df['url']  


columns_to_keep = ['id', 'title', 'selftext', 'author', 'score', 
                   'num_comments', 'created_utc', 'subreddit', 'url', 'permalink']


if 'author' not in politics_df.columns:
    politics_df['author'] = 'unknown'


politics_df = politics_df[columns_to_keep]


cursor.execute('''
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    title TEXT,
    selftext TEXT,
    author TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_utc INTEGER,
    subreddit TEXT,
    url TEXT,
    permalink TEXT
)
''')


politics_df.to_sql('posts', conn, if_exists='replace', index=False)

print(f" Successfully loaded {len(politics_df)} posts into database!")

#Verify
cursor.execute("SELECT COUNT(*) FROM posts")
count = cursor.fetchone()[0]
print(f" Database contains {count} posts")

cursor.execute("SELECT title, score, num_comments FROM posts LIMIT 5")
print("\nSample posts:")
for title, score, comments in cursor.fetchall():
    print(f"  - {title[:50]}... (score: {score}, comments: {comments})")

conn.close()
print("\n Ready to run analysis! Now execute: python analyze_reddit_data.py")







