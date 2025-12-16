import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import os

def run_topic_modeling():
    print("--- Starting Topic Modeling (LDA) ---")

    # --- 1. SMART PATH SETUP ---
    # Finds the database automatically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_db_paths = [
        os.path.join(current_dir, 'reddit_data.db'),
        os.path.join(current_dir, '..', 'reddit_data.db'),
        os.path.join(current_dir, 'reddit-political-analysis', 'reddit_data.db')
    ]
    
    db_path = None
    project_root = None
    
    for path in possible_db_paths:
        if os.path.exists(path):
            db_path = path
            project_root = os.path.dirname(db_path)
            print(f"Found database at: {db_path}")
            break
            
    if db_path is None:
        print("ERROR: Could not find 'reddit_data.db'.")
        print("Please run 'load_data.py' first to create the database.")
        return

    # Use 'output' (Singular) to match your folder structure
    output_dir = os.path.join(project_root, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. LOAD DATA ---
    conn = sqlite3.connect(db_path)
    
    # UPDATED: We specifically select the 'title' column here
    try:
        df = pd.read_sql_query("SELECT title FROM posts", conn)
        print(f"Loaded {len(df)} posts from database.")
    except Exception as e:
        print(f"Error reading from database: {e}")
        conn.close()
        return
        
    conn.close()

    # --- 3. PREPROCESSING ---
    print("Vectorizing text (using titles)...")
    
    # Add common political/reddit stop words to ignore
    my_stop_words = list(CountVectorizer(stop_words='english').get_stop_words())
    my_stop_words.extend(['reddit', 'comments', 'breaking', 'discussion', 'politics', 'megathread', 'thread', 'post'])

    # Vectorize the 'title' column
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=my_stop_words)
    tf = tf_vectorizer.fit_transform(df['title'])

    # --- 4. RUN LDA ---
    n_topics = 5
    print(f"Running LDA to find {n_topics} topics...")
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20, learning_method='online', random_state=42)
    lda.fit(tf)

    # --- 5. VISUALIZE ---
    print("Generating topic chart...")
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [tf_feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color='teal')
        ax.set_title(f'Topic {topic_idx +1}', fontdict={'fontsize': 14})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.suptitle('Top Words per Topic (LDA Model)', fontsize=18)
    plt.tight_layout()
    
    # Save to output folder
    save_path = os.path.join(output_dir, 'topic_model_results.png')
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    run_topic_modeling()