"""
Airflow DAG: Reddit Political Discourse Analysis Pipeline

This replaces the original four manual scripts (load_data.py, analyze_reddit_data.py,
advanced_sentiment.py, topic_modeling.py) with one orchestrated pipeline. Each function
below decorated with @task is one Airflow task. Airflow schedules them, retries them on
failure, and shows the dependency graph and per-task logs in its UI.

HOW TO RUN THIS LOCALLY
1. python -m venv airflow_venv && source airflow_venv/bin/activate
2. pip install -r requirements-airflow.txt
3. python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
4. export AIRFLOW_HOME=~/airflow
5. export REDDIT_PROJECT_ROOT=/path/to/reddit-political-analysis   (optional; see below)
6. airflow standalone   (sets up the metadata DB, creates a login, starts webserver + scheduler)
7. cp dags/reddit_political_analysis_dag.py $AIRFLOW_HOME/dags/
8. Open localhost:8080, find "reddit_political_analysis", and trigger it manually

WHERE THIS DAG LOOKS FOR THE REPO
The pipeline reads data/politics.csv and writes reddit_data.db + output/ inside the repo.
PROJECT_ROOT is resolved automatically, in this order:
  1. $REDDIT_PROJECT_ROOT if it is set and points at a real directory
  2. walking up from this file's location until a folder containing data/politics.csv
     is found (works when the DAG is run from inside the repo's dags/ folder)
  3. ~/Downloads/reddit-political-analysis as a last-resort default
When Airflow parses the copy in $AIRFLOW_HOME/dags/ (outside the repo), option 2 cannot
work, so set REDDIT_PROJECT_ROOT in the environment you launch `airflow standalone` from.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task


# ---------------------------------------------------------------------------
# Resolve the repo location on disk (see module docstring for the rules)
# ---------------------------------------------------------------------------
def _detect_project_root() -> str:
    env_root = os.environ.get("REDDIT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    here = os.path.dirname(os.path.abspath(__file__))
    current = here
    while True:
        if os.path.exists(os.path.join(current, "data", "politics.csv")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return os.path.expanduser("~/Downloads/reddit-political-analysis")


PROJECT_ROOT = _detect_project_root()
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "politics.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "reddit_data.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

default_args = {
    "owner": "landon",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    dag_id="reddit_political_analysis",
    schedule=None,  # trigger manually from the UI; swap for "@daily" to run on a schedule
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["reddit", "nlp", "portfolio"],
)
def reddit_political_analysis():
    """DAG entrypoint. Airflow runs this once, at parse time, to build the task graph
    defined below — it does not re-run this function's body on every task execution."""

    @task
    def load_data() -> int:
        """Task 1: read the Kaggle CSV and load it into SQLite as the 'posts' table.

        Note: the original load_data.py had a dedent partway through that made the
        rest of the script run at module level instead of inside the function, so it
        referenced variables (politics_df, conn) that no longer existed and would
        raise a NameError before finishing. This version keeps the same intended
        logic — rename columns, add subreddit/permalink, load to SQLite — as one
        clean function.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={
            "comms_num": "num_comments",
            "created": "created_utc",
            "body": "selftext",
        })
        df["subreddit"] = "politics"
        df["permalink"] = df["url"]
        if "author" not in df.columns:
            df["author"] = "unknown"

        columns_to_keep = ["id", "title", "selftext", "author", "score",
                            "num_comments", "created_utc", "subreddit", "url", "permalink"]
        df = df[columns_to_keep]

        conn = sqlite3.connect(DB_PATH)
        df.to_sql("posts", conn, if_exists="replace", index=False)
        conn.close()

        print(f"Loaded {len(df)} posts into {DB_PATH}")
        return len(df)  # small value, safe to pass through Airflow's XCom

    @task
    def run_vader_sentiment(row_count: int) -> str:
        """Task 2: VADER sentiment scoring, written to a new 'posts_scored' table.

        Takes row_count as an argument purely so Airflow knows this must run after
        load_data — the number itself isn't used. This task re-reads from SQLite
        rather than having load_data hand off the full DataFrame, since XCom (the
        mechanism Airflow uses to pass task outputs) is meant for small values, not
        entire datasets.
        """
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM posts", conn)
        conn.close()

        df["date"] = pd.to_datetime(df["created_utc"], unit="s")

        sia = SentimentIntensityAnalyzer()
        scores = df["title"].fillna("").apply(sia.polarity_scores)
        df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
        df["sentiment_category"] = df["sentiment_compound"].apply(
            lambda c: "Positive" if c > 0.05 else ("Negative" if c < -0.05 else "Neutral")
        )

        conn = sqlite3.connect(DB_PATH)
        df.to_sql("posts_scored", conn, if_exists="replace", index=False)
        conn.close()

        print(f"Scored {len(df)} posts with VADER")
        return "posts_scored"

    @task
    def generate_visualizations(table_name: str) -> None:
        """Task 3: sentiment distribution, timeline, top keywords, and engagement charts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import nltk
        from nltk.corpus import stopwords

        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords")

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        df["date"] = pd.to_datetime(df["created_utc"], unit="s")

        # sentiment distribution
        counts = df["sentiment_category"].value_counts()
        plt.figure(figsize=(10, 6))
        plt.bar(counts.index, counts.values, color=["green", "gray", "red"])
        plt.title("Sentiment Distribution of Reddit Posts")
        plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # sentiment over time
        daily = df.sort_values("date").groupby(df["date"].dt.date)["sentiment_compound"].mean()
        plt.figure(figsize=(14, 6))
        plt.plot(daily.index, daily.values, marker="o", linewidth=2)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("Sentiment Trend Over Time")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_timeline.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # top keywords
        stop_words = set(stopwords.words("english"))
        all_text = " ".join(df["title"].fillna("").astype(str)).lower()
        words = re.findall(r"\b[a-z]{3,}\b", all_text)
        filtered = [w for w in words if w not in stop_words and w not in ["https", "http", "www", "reddit", "com"]]
        top = Counter(filtered).most_common(20)
        top_words, top_counts = zip(*top)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_words)), top_counts, color="steelblue")
        plt.yticks(range(len(top_words)), top_words)
        plt.title("Top 20 Keywords")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_keywords.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # engagement (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(df["score"], bins=50, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Distribution of Post Scores")

        top_posts = df.nlargest(10, "score")[["title", "score"]]
        axes[0, 1].barh(range(len(top_posts)), top_posts["score"], color="coral")
        axes[0, 1].set_yticks(range(len(top_posts)))
        axes[0, 1].set_yticklabels([t[:40] for t in top_posts["title"]], fontsize=8)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title("Top 10 Posts by Score")

        axes[1, 0].scatter(df["sentiment_compound"], df["score"], alpha=0.3, color="purple")
        z = np.polyfit(df["sentiment_compound"].fillna(0), df["score"].fillna(0), 1)
        axes[1, 0].plot(df["sentiment_compound"], np.poly1d(z)(df["sentiment_compound"]), "r--")
        axes[1, 0].set_title("Sentiment vs Engagement")

        daily_posts = df.sort_values("date").groupby(df["date"].dt.date).size()
        axes[1, 1].plot(daily_posts.index, daily_posts.values, marker="o", color="green")
        axes[1, 1].set_title("Posting Activity Over Time")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "engagement_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved 4 charts to", OUTPUT_DIR)

    @task
    def generate_summary_report(table_name: str) -> None:
        """Task 4: writes a text summary to output/summary_report.txt instead of just
        printing it, so it's still there after the task finishes."""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()

        lines = [f"Total posts analyzed: {len(df)}"]
        pct = df["sentiment_category"].value_counts(normalize=True) * 100
        for sentiment, p in pct.items():
            lines.append(f"  {sentiment}: {p:.1f}%")
        lines.append(f"Average sentiment score: {df['sentiment_compound'].mean():.3f}")
        lines.append(f"Average post score: {df['score'].mean():.1f}")

        report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Report written to {report_path}")

    @task
    def run_roberta_sentiment() -> None:
        """Task 5 (parallel branch): RoBERTa sentiment via Hugging Face transformers.
        Only needs the raw 'posts' table, so it doesn't need to wait on run_vader_sentiment
        and can run alongside it instead."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from transformers import pipeline

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT title FROM posts", conn)
        conn.close()

        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1)
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

        def get_sentiment(text: str) -> str:
            try:
                result = sentiment_task(text[:512])[0]
                return label_map.get(result["label"], "Neutral")
            except Exception:
                return "Neutral"

        df["sentiment_label"] = df["title"].apply(get_sentiment)
        df.to_csv(os.path.join(OUTPUT_DIR, "roberta_sentiment_results.csv"), index=False)

        plt.figure(figsize=(10, 6))
        sns.countplot(x="sentiment_label", data=df, order=["Negative", "Neutral", "Positive"])
        plt.title("Sentiment Distribution (RoBERTa)")
        plt.savefig(os.path.join(OUTPUT_DIR, "roberta_sentiment_chart.png"))
        plt.close()

        print(f"RoBERTa-scored {len(df)} posts")

    @task
    def run_topic_modeling() -> None:
        """Task 6 (parallel branch): LDA topic modeling. Also only needs 'posts', so it
        runs in parallel with run_vader_sentiment and run_roberta_sentiment."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT title FROM posts", conn)
        conn.close()

        extra_stop_words = ["reddit", "comments", "breaking", "discussion",
                             "politics", "megathread", "thread", "post"]
        stop_words = list(CountVectorizer(stop_words="english").get_stop_words()) + extra_stop_words

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
        tf = vectorizer.fit_transform(df["title"])

        lda = LatentDirichletAllocation(n_components=5, max_iter=20, learning_method="online", random_state=42)
        lda.fit(tf)

        feature_names = vectorizer.get_feature_names_out()
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        for topic_idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_idx]
            weights = topic[top_idx]
            axes[topic_idx].barh(top_words, weights, color="teal")
            axes[topic_idx].set_title(f"Topic {topic_idx + 1}")
            axes[topic_idx].invert_yaxis()

        plt.suptitle("Top Words per Topic (LDA)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "topic_model_results.png"))
        plt.close()

        print("Saved LDA topic model chart")

    # --- task graph ---
    # load_data runs first. run_vader_sentiment runs after it and feeds both
    # generate_visualizations and generate_summary_report. run_roberta_sentiment and
    # run_topic_modeling only need the raw 'posts' table, so they run in parallel with
    # run_vader_sentiment rather than waiting for it.
    row_count = load_data()
    scored_table = run_vader_sentiment(row_count)
    generate_visualizations(scored_table)
    generate_summary_report(scored_table)

    roberta_task = run_roberta_sentiment()
    topic_task = run_topic_modeling()
    row_count >> [roberta_task, topic_task]


reddit_political_analysis()
