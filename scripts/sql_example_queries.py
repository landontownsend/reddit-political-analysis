"""
SQL Query Examples
Demonstrates various SQL queries for analyzing Reddit political data
"""

import sqlite3
import pandas as pd
import os

#Get database path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, 'reddit_data.db')

conn = sqlite3.connect(db_path)

print("="*70)
print("SQL QUERY DEMONSTRATIONS")
print("="*70)

#Query 1: Basic aggregation
print("\n1. TOP 10 HIGHEST SCORING POSTS")
print("-" * 70)
query1 = """
SELECT title, score, num_comments, author
FROM posts
ORDER BY score DESC
LIMIT 10;
"""
df1 = pd.read_sql_query(query1, conn)
print(df1.to_string(index=False))

#Query 2: Average scores
print("\n\n2. ENGAGEMENT STATISTICS")
print("-" * 70)
query2 = """
SELECT 
    COUNT(*) as total_posts,
    AVG(score) as avg_score,
    MAX(score) as max_score,
    MIN(score) as min_score,
    AVG(num_comments) as avg_comments
FROM posts;
"""
df2 = pd.read_sql_query(query2, conn)
print(df2.to_string(index=False))

#Query 3: Posts by score range
print("\n\n3. POSTS GROUPED BY SCORE RANGES")
print("-" * 70)
query3 = """
SELECT 
    CASE 
        WHEN score < 10 THEN 'Low (0-9)'
        WHEN score < 100 THEN 'Medium (10-99)'
        WHEN score < 1000 THEN 'High (100-999)'
        ELSE 'Viral (1000+)'
    END as score_category,
    COUNT(*) as post_count,
    AVG(num_comments) as avg_comments
FROM posts
GROUP BY score_category
ORDER BY MIN(score);
"""
df3 = pd.read_sql_query(query3, conn)
print(df3.to_string(index=False))
