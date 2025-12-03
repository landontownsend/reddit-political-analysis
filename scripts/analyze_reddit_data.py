import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import numpy as np

#Download required NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

class RedditAnalyzer:
    def __init__(self, db_path='reddit_data.db'):
        """Initialize analyzer with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        #Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def get_posts_dataframe(self):
        """Load posts from database"""
        query = "SELECT * FROM posts"
        df = pd.read_sql_query(query, self.conn)
        
        #Convert timestamp to datetime
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        
        return df
    
    def analyze_sentiment(self, df, text_column='title'):
        """Add sentiment scores to dataframe"""
        print(f"Analyzing sentiment for {len(df)} items...")
        
        #Ensure text column exists and has no null values
        if text_column not in df.columns:
            print(f"Warning: {text_column} column not found")
            return df
        
        df[text_column] = df[text_column].fillna('')
        
        #Calculate sentiment scores
        sentiments = []
        for text in df[text_column]:
            try:
                scores = self.sia.polarity_scores(str(text))
                sentiments.append(scores)
            except:
                sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
        
        #Add sentiment columns
        df['sentiment_neg'] = [s['neg'] for s in sentiments]
        df['sentiment_neu'] = [s['neu'] for s in sentiments]
        df['sentiment_pos'] = [s['pos'] for s in sentiments]
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        
        #Categorize sentiment
        df['sentiment_category'] = df['sentiment_compound'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )
        
        return df
    
    def plot_sentiment_distribution(self, df, output_file='sentiment_distribution.png'):
        """Plot sentiment distribution"""
        if 'sentiment_category' not in df.columns:
            print("No sentiment data available")
            return
        
        plt.figure(figsize=(10, 6))
        
        #Count sentiment categories
        sentiment_counts = df['sentiment_category'].value_counts()
        
        #Create bar plot
        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[colors.get(x, 'blue') for x in sentiment_counts.index])
        
        plt.title('Sentiment Distribution of Reddit Posts', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment distribution to {output_file}")
        plt.close()
    
    def plot_sentiment_over_time(self, df, output_file='sentiment_timeline.png'):
        """Plot sentiment trends over time"""
        if 'date' not in df.columns or 'sentiment_compound' not in df.columns:
            print("Missing date or sentiment data")
            return
        
        plt.figure(figsize=(14, 6))
        
        #Group by date and calculate average sentiment
        df_sorted = df.sort_values('date')
        df_sorted['date_only'] = df_sorted['date'].dt.date
        
        daily_sentiment = df_sorted.groupby('date_only')['sentiment_compound'].agg(['mean', 'count'])
        
        #Plot
        plt.plot(daily_sentiment.index, daily_sentiment['mean'], 
                marker='o', linewidth=2, markersize=4, alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        
        plt.title('Sentiment Trend Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment timeline to {output_file}")
        plt.close()
    
    def analyze_top_keywords(self, df, text_column='title', top_n=20):
        """Extract and count most common keywords"""
        if text_column not in df.columns:
            print(f"Column {text_column} not found")
            return Counter()
        
        #Combine all text
        all_text = ' '.join(df[text_column].fillna('').astype(str))
        
        #Clean and tokenize
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        
        #Remove stopwords/common reddit terms
        filtered_words = [w for w in words if w not in self.stop_words 
                         and w not in ['https', 'http', 'www', 'reddit', 'com']]
        
        #Count and return top keywords
        return Counter(filtered_words).most_common(top_n)
    
    def plot_top_keywords(self, df, output_file='top_keywords.png', top_n=20):
        """Plot most common keywords"""
        keywords = self.analyze_top_keywords(df, top_n=top_n)
        
        if not keywords:
            print("No keywords found")
            return
        
        words, counts = zip(*keywords)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), counts, color='steelblue')
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Common Keywords', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved top keywords to {output_file}")
        plt.close()
    
    def analyze_engagement(self, df, output_file='engagement_analysis.png'):
        """Analyze post engagement metrics"""
        if 'score' not in df.columns:
            print("No score data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Score distribution
        axes[0, 0].hist(df['score'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Post Scores', fontweight='bold')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Top posts by score
        top_posts = df.nlargest(10, 'score')[['title', 'score']]
        axes[0, 1].barh(range(len(top_posts)), top_posts['score'], color='coral')
        axes[0, 1].set_yticks(range(len(top_posts)))
        axes[0, 1].set_yticklabels([t[:50] + '...' if len(t) > 50 else t 
                                    for t in top_posts['title']], fontsize=8)
        axes[0, 1].set_title('Top 10 Posts by Score', fontweight='bold')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].invert_yaxis()
        
        # 3. Sentiment vs Score
        if 'sentiment_compound' in df.columns:
            axes[1, 0].scatter(df['sentiment_compound'], df['score'], 
                             alpha=0.3, color='purple')
            axes[1, 0].set_title('Sentiment vs Engagement', fontweight='bold')
            axes[1, 0].set_xlabel('Sentiment Score')
            axes[1, 0].set_ylabel('Post Score')
            
            
            z = np.polyfit(df['sentiment_compound'].fillna(0), df['score'].fillna(0), 1)
            p = np.poly1d(z)
            axes[1, 0].plot(df['sentiment_compound'], p(df['sentiment_compound']), 
                          "r--", alpha=0.8, linewidth=2)
        
        # 4. Posts over time
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
            df_sorted['date_only'] = df_sorted['date'].dt.date
            daily_posts = df_sorted.groupby('date_only').size()
            
            axes[1, 1].plot(daily_posts.index, daily_posts.values, 
                          marker='o', linewidth=2, color='green')
            axes[1, 1].set_title('Posting Activity Over Time', fontweight='bold')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Number of Posts')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved engagement analysis to {output_file}")
        plt.close()
    
    def generate_summary_report(self, df):
        """Generate text summary of analysis"""
        print("\n" + "="*60)
        print("REDDIT POLITICAL ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        #Basic statistics
        print(f"\nDataset Overview:")
        print(f"  Total posts analyzed: {len(df)}")
        
        if 'date' in df.columns:
            print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        if 'subreddit' in df.columns:
            print(f"  Subreddits: {', '.join(df['subreddit'].unique())}")
        
        #Sentiment analysis
        if 'sentiment_category' in df.columns:
            print(f"\nSentiment Breakdown:")
            sentiment_pct = df['sentiment_category'].value_counts(normalize=True) * 100
            for sentiment, pct in sentiment_pct.items():
                print(f"  {sentiment}: {pct:.1f}%")
            
            avg_sentiment = df['sentiment_compound'].mean()
            print(f"\n  Average sentiment score: {avg_sentiment:.3f}")
            print(f"  Overall tone: {'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'}")
        
        #Engagement statistics
        if 'score' in df.columns:
            print(f"\nEngagement Metrics:")
            print(f"  Average score: {df['score'].mean():.1f}")
            print(f"  Median score: {df['score'].median():.1f}")
            print(f"  Highest scoring post: {df['score'].max()}")
        
        #Top keywords
        print(f"\nTop 10 Keywords:")
        keywords = self.analyze_top_keywords(df, top_n=10)
        for i, (word, count) in enumerate(keywords, 1):
            print(f"  {i}. {word}: {count}")
        
        print("\n" + "="*60)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Reddit Political Analysis...")
        print("="*60)
        
        #Load data
        print("\n1. Loading data from database...")
        posts_df = self.get_posts_dataframe()
        print(f"   Loaded {len(posts_df)} posts")
        
        #Analyze sentiment
        print("\n2. Analyzing sentiment...")
        posts_df = self.analyze_sentiment(posts_df, text_column='title')
        
        #Generate visualizations
        print("\n3. Generating visualizations...")
        self.plot_sentiment_distribution(posts_df)
        self.plot_sentiment_over_time(posts_df)
        self.plot_top_keywords(posts_df)
        self.analyze_engagement(posts_df)
        
        #Generate summary
        print("\n4. Generating summary report...")
        self.generate_summary_report(posts_df)
        
        print("\n✓ Analysis complete! Check the output files.")
        
        self.conn.close()

if __name__ == "__main__":
    analyzer = RedditAnalyzer()
    analyzer.run_full_analysis()
