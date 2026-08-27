# Reddit Political Analysis Project

A comprehensive sentiment analysis and discourse examination of political discussions on Reddit's r/politics subreddit using Python, SQLite, and natural language processing.

## Project Overview

This project analyzes political discourse patterns on Reddit by examining sentiment trends, keyword frequencies, and engagement metrics from r/politics discussions. The analysis uses VADER sentiment analysis, SQL queries for data management, and statistical visualizations to uncover patterns in political discourse.

### Key Features

- **Sentiment Analysis**: VADER-based sentiment scoring of post titles
- **SQL Database Management**: Efficient data storage and querying with SQLite
- **Keyword Extraction**: Identification of most frequently discussed political topics
- **Engagement Analytics**: Analysis of upvotes, comments, and posting patterns
- **Temporal Analysis**: Sentiment and activity trends over time
- **Data Visualization**: Professional charts and graphs for presenting findings
- **Topic Modeling**: Modeling of topics using LDA Analysis
- **RoBERTa Analysis**: Sentiment analysis using RoBERTa neural model

## Project Structure

```
reddit-political-analysis/
├── data/
│   ├── politics.csv              # Reddit dataset (user must download)
│   └── .gitkeep                  # Tracks folder structure
├── scripts/
│   ├── load_data.py              # Loads CSV data into SQLite database
│   ├── analyze_reddit_data.py    # Main analysis and visualization script (VADER)
│   ├── advanced_sentiment.py     # RoBERTa (transformers) sentiment analysis
│   ├── topic_modeling.py         # LDA topic modeling
│   └── sql_example_queries.py    # SQL query demonstrations (optional)
├── dags/
│   └── reddit_political_analysis_dag.py  # Airflow DAG: all of the above as one pipeline
├── output/                       # Generated visualizations (created automatically)
│   ├── sentiment_distribution.png
│   ├── sentiment_timeline.png
│   ├── top_keywords.png
│   ├── engagement_analysis.png
│   ├── roberta_sentiment_chart.png
│   ├── roberta_sentiment_results.csv
│   ├── topic_model_results.png
│   └── summary_report.txt        # Text summary (written by the Airflow pipeline)
├── reddit_data.db               # SQLite database (generated)
├── requirements.txt             # Python dependencies for the standalone scripts
├── requirements-airflow.txt     # Extra dependencies for the Airflow pipeline (heavy)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip 
- Git
- Free Kaggle account (for dataset download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/landontownsend/reddit-political-analysis.git
cd reddit-political-analysis
```

### Step 2: Download the Dataset

1. Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/thedevastator/analyzing-the-political-discourse-of-reddit-s-su)
2. Click the **Download** button (requires free Kaggle account)
3. Extract the downloaded ZIP file
4. Place `politics.csv` in the `data/` folder of this project

**Expected file location:**
```
reddit-political-analysis/data/politics.csv
```

### Step 3: Create Virtual Environment

```bash
#Create virtual environment
python -m venv venv

# Activate virtual environment
#On Mac/Linux:
source venv/bin/activate
#On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
```

## Usage

### Load Data into Database

```bash
cd scripts
python load_data.py
```

This script will:
- Read the CSV file from the `data/` folder
- Process and clean the data
- Create a SQLite database (`reddit_data.db`)
- Load all posts into structured tables
- Display summary statistics

### Run Analysis

```bash
python analyze_reddit_data.py
```

This will:
- Analyze sentiment of all posts using VADER
- Extract top keywords from discussions
- Generate 4 visualization files in the `outputs/` folder
- Print a comprehensive summary report to console

```bash
python advanced_sentiment.py
```

This will:
- Perform RoBERTa sentiment analysis
- Generate visualization in the `outputs/` folder

```bash
python topic_modeling.py
```

This will:
- Conduct topic modeling according to the LDA model
- Produce visualization displaying 5 topics in the `outputs/` folder

### Run SQL Query Demonstrations (Optional)

```bash
python sql_queries.py
```

Displays various SQL query examples demonstrating database operations.

## Output Files

After running the analysis, you'll find these visualizations in the `outputs/` folder:

1. **sentiment_distribution.png** - Bar chart showing distribution of positive, negative, and neutral posts
2. **sentiment_timeline.png** - Line graph of sentiment trends over time
3. **top_keywords.png** - Horizontal bar chart of most frequently discussed topics
4. **engagement_analysis.png** - Multi-panel visualization showing:
   - Distribution of post scores
   - Top 10 highest-scoring posts
   - Sentiment vs. engagement scatter plot
   - Daily posting activity
5. **roberta_sentiment_chart.png** - Bar chart visualising sentiment classiifcation results from RoBERTa model.
6. **topic_model_results.png** - Displays results from LDA topic modeling analysis.

The Airflow pipeline (below) produces all of the above plus `roberta_sentiment_results.csv`
and a `summary_report.txt` text summary.

## Airflow Orchestration

The four manual steps (`load_data.py`, `analyze_reddit_data.py`, `advanced_sentiment.py`,
`topic_modeling.py`) are also available as a single orchestrated **Apache Airflow** pipeline
defined with the TaskFlow API in [`dags/reddit_political_analysis_dag.py`](dags/reddit_political_analysis_dag.py).
Airflow runs the tasks in dependency order, retries them on failure, and gives you a UI with
the dependency graph and per-task logs.

### Pipeline (6 tasks)

```
load_data ──> run_vader_sentiment ──> generate_visualizations
        │                        └──> generate_summary_report
        ├──> run_roberta_sentiment
        └──> run_topic_modeling
```

| Task | Replaces | Output |
|------|----------|--------|
| `load_data` | `load_data.py` | `posts` table in `reddit_data.db` |
| `run_vader_sentiment` | VADER part of `analyze_reddit_data.py` | `posts_scored` table |
| `generate_visualizations` | charts in `analyze_reddit_data.py` | `sentiment_distribution.png`, `sentiment_timeline.png`, `top_keywords.png`, `engagement_analysis.png` |
| `generate_summary_report` | console summary in `analyze_reddit_data.py` | `output/summary_report.txt` |
| `run_roberta_sentiment` | `advanced_sentiment.py` | `roberta_sentiment_chart.png`, `roberta_sentiment_results.csv` |
| `run_topic_modeling` | `topic_modeling.py` | `topic_model_results.png` |

`run_roberta_sentiment` and `run_topic_modeling` only need the raw `posts` table, so they run
in parallel with `run_vader_sentiment` instead of waiting for it.

### Running it locally

A helper script does the venv + install + NLTK download (and, on macOS, the
`setproctitle` workaround described below):

```bash
./scripts/setup_airflow.sh
source airflow_venv/bin/activate

export AIRFLOW_HOME=~/airflow
export REDDIT_PROJECT_ROOT=$(pwd)     # so the DAG finds this repo on disk
airflow standalone                    # first run creates the DB + an admin login

# in another shell (same env): install the DAG and trigger a run
cp dags/reddit_political_analysis_dag.py "$AIRFLOW_HOME/dags/"
airflow dags trigger reddit_political_analysis
# ...or open http://localhost:8080 and trigger "reddit_political_analysis" from the UI
```

Doing it by hand instead:

```bash
python3.12 -m venv airflow_venv && source airflow_venv/bin/activate
AIRFLOW_VERSION=2.10.4; PYTHON_VERSION=3.12
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install -r requirements-airflow.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

`PROJECT_ROOT` in the DAG is resolved automatically: it uses `$REDDIT_PROJECT_ROOT` if set,
otherwise it walks up from the DAG file until it finds `data/politics.csv`, otherwise it
falls back to `~/Downloads/reddit-political-analysis`. Set `REDDIT_PROJECT_ROOT` before
`airflow standalone` when the copy in `$AIRFLOW_HOME/dags/` lives outside the repo.

### macOS note

On Apple-silicon macOS, Airflow's forking task runner can hang inside `setproctitle`
(CoreFoundation is not `fork()`-safe), leaving tasks stuck in `running`. `scripts/setup_airflow.sh`
installs a small `.pth` hook in the venv that makes `setproctitle` a no-op, which fixes it.
If the webserver's log workers crash (`Worker ... was sent SIGSEGV!`), run the components
separately instead of `airflow standalone`:

```bash
airflow scheduler --skip-serve-logs &
airflow webserver --port 8080 &
```

## Database Schema

The project uses SQLite for efficient data storage and querying. For detailed SQL query examples and demonstrations, see `scripts/sql_queries.py`.

### Posts Table

```sql
CREATE TABLE posts (
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
);
```

## Technologies Used

- **Python 3.8+** - Core programming language
- **pandas** - Data manipulation and analysis
- **SQLite3** - Relational database for data storage
- **NLTK (VADER)** - Sentiment analysis
- **matplotlib** - Data visualization
- **seaborn** - Enhanced plot styling
- **NumPy** - Numerical computations
- **transformers (Hugging Face)** - Advanced sentiment analysis (RoBERTa model)
- **scikit-learn** - Topic Modeling algorithims (LDA)
- **Apache Airflow** - Pipeline orchestration (TaskFlow API)

## Analysis Methodology

### Data Collection
- Dataset sourced from Kaggle containing r/politics posts from December 2022
- Posts include titles, scores, comment counts, timestamps, and URLs

### Sentiment Analysis
- Uses VADER (Valence Aware Dictionary and Sentiment Reasoner)
- Assigns compound sentiment scores from -1 to +1
- Categorizes posts as Positive (>0.05), Negative (<-0.05), or Neutral

### RoBERTa Neural Networking
- Uses Transformer-based neural network.
- Performs advanced sentiment classification on posts.
- Implemented using Hugging Face transformers library.

### Keyword Extraction
- Tokenizes post titles using regex
- Removes common stop words and Reddit-specific terms
- Counts and ranks word frequencies

### Topic Modeling
- This module implements unsupervised machine learning using Latent Dirichlet Allocation (LDA) to uncover thematic clusters within discourse.
- Key outputs include interpretably grouped keywords representing distinct topics, providing insight into the underlying narratives of the analyzed data.

### Engagement Metrics
- Analyzes relationship between sentiment and post scores
- Identifies temporal patterns in posting activity
- Examines distribution of engagement levels

## Sample Findings

The analysis reveals insights such as:
- Distribution of sentiment in political discourse
- Most frequently discussed political topics
- Correlation between sentiment and engagement
- Temporal patterns in posting behavior

*Note: Specific findings depend on the dataset time period analyzed*

## Academic Use

This project is designed for academic research and educational purposes. When using this project ensure the following:

1. **Cite the dataset**: Credit the original Kaggle data source
2. **Follow ethical guidelines**: Respect user privacy (no individual user identification)
3. **Acknowledge limitations**: Note the specific time period and subreddit analyzed
4. **Consider bias**: Reddit demographics may not represent general population

## Customization (with SQL)

### Custom SQL Queries

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('reddit_data.db')

# Your custom query
custom_query = """
SELECT title, score 
FROM posts 
WHERE score > 1000
ORDER BY score DESC
"""

results = pd.read_sql_query(custom_query, conn)
print(results)

conn.close()
```

## Troubleshooting

### Issue: "CSV file not found"

**Solution**: Ensure `politics.csv` is in the `data/` folder:
```bash
ls data/politics.csv
```

### Issue: "Module not found" errors

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: NLTK data not found

**Solution**: Download required NLTK datasets:
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

### Issue: Database locked error

**Solution**: Close any other programs accessing the database, then retry.

### Issue: Empty visualizations

**Solution**: Verify data loaded correctly:
```bash
sqlite3 reddit_data.db "SELECT COUNT(*) FROM posts;"
```

## Future Enhancements

Potential improvements for this project:

- [x] Topic modeling using Latent Dirichlet Allocation (`scripts/topic_modeling.py`, `run_topic_modeling` task)
- [x] Orchestration with Apache Airflow (`dags/reddit_political_analysis_dag.py`)
- [ ] Network analysis of user interactions
- [ ] Comparison across multiple political subreddits
- [ ] Time-series forecasting of sentiment trends over larger time period
- [ ] Word cloud visualizations
- [ ] Export results to PDF report
- [ ] Web dashboard for interactive exploration
- [ ] Real-time analysis with Reddit API (with the condition that the API is made self-service again)

## Permissions

This project is open source and available for educational use. When using or building upon this work:
- Credit the original dataset from Kaggle
- Acknowledge this repository if used as a foundation
- Follow Reddit's Terms of Service and API guidelines (if expanding and incorporating the official Reddit API)

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

**Landon Townsend**

Email: landontownsend20@gmail.com

GitHub: [@landontownsend](https://github.com/landontownsend)

Project Link: [https://github.com/landontownsend/reddit-political-analysis](https://github.com/landontownsend/reddit-political-analysis)

## Acknowledgments

- Reddit and r/politics community for the discourse data
- Kaggle user for compiling and sharing the dataset
- NLTK team for the VADER sentiment analysis tool
- Python data science community for excellent libraries

---

**Dataset Time Period**: December 2022  
**Last Updated**: August 2026  
**Python Version**: 3.8+

---

*This project was created for educational purposes to demonstrate data analysis, SQL database management, sentiment analysis, and data visualization techniques.*
