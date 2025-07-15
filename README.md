

# MovieLens Advanced Recommendation System

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-%E2%9C%94%EF%B8%8F-brightgreen)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg)](https://grouplens.org/datasets/movielens/)

## Project Overview

In the era of endless streaming content, users are often overwhelmed by the sheer number of movie choices available. This project addresses that challenge by building an advanced, production-ready movie recommendation system using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. The system combines collaborative filtering, content-based filtering, and state-of-the-art ensemble machine learning models (XGBoost, LightGBM, CatBoost) to deliver highly accurate, personalized movie recommendations.

## Features

- **Multiple Recommendation Approaches:**
  - Content-based filtering (genres, tags)
  - User-based collaborative filtering
  - Item-based collaborative filtering
  - SVD-based matrix factorization
  - Hybrid and ensemble models (XGBoost, LightGBM, CatBoost)
- **Streamlit Web App:**
  - Interactive UI for exploring recommendations, analytics, and model performance
  - Visualizations: rating distributions, genre popularity, user activity, model comparisons
  - A/B testing simulator for business impact analysis
- **Comprehensive Evaluation:**
  - Metrics: RMSE, MAE, Precision@K, Recall@K, MAP, NDCG
  - Business-focused insights for deployment and user engagement

## Data Description

- **Source:** [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)
- **Size:** 100,836 ratings, 3,683 tags, 9,742 movies, 610 users
- **Files:**
  - `movies.csv`: Movie metadata (title, genres)
  - `ratings.csv`: User ratings (userId, movieId, rating, timestamp)
  - `tags.csv`: User-generated tags
  - `links.csv`: External links to IMDb and TMDb
- **Preprocessing:**
  - Merging, cleaning, and feature engineering (genres, tags, user/movie stats)
  - Removal of users/movies with <5 ratings to reduce sparsity
  - One-hot encoding, normalization, and timestamp conversion

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd datascience
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv ds-env
   # On Windows:
   ds-env\Scripts\activate
   # On Unix/Mac:
   source ds-env/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

- Open the provided local URL in your browser to interact with the MovieLens Recommendation System.
- Explore recommendations, analytics, and model performance through the web interface.

### Jupyter Notebook

- The full modeling pipeline, data exploration, and evaluation are documented in `index.ipynb`.
- Open the notebook in JupyterLab or VSCode to review and experiment with the code.

## Methodology

1. **Data Preprocessing:**
   - Load and merge MovieLens data files
   - Clean missing values, engineer features (genres, tags, user/movie stats)
   - Normalize ratings, reduce sparsity
2. **Model Development:**
   - Content-based filtering (TF-IDF + Nearest Neighbors)
   - User-based and item-based collaborative filtering
   - SVD-based matrix factorization
   - Advanced ensemble models (XGBoost, LightGBM, CatBoost)
   - Hybrid and stacking approaches
3. **Evaluation:**
   - Metrics: RMSE, MAE, Precision@K, Recall@K, MAP, NDCG
   - Business impact: A/B testing simulator, engagement metrics

## Results

- **SVD-based Collaborative Filtering:**
  - Precision@5: 80%
  - RMSE: 0.94
- **Hybrid Ensemble (XGBoost + SVD):**
  - Precision@10: 90%
  - RMSE: 0.80
- **Business Impact:**
  - Demonstrated improvements in simulated click-through, conversion, and engagement rates
- **Scalability:**
  - Efficient for large user-item matrices; handles cold-start scenarios

## License & Citation

- **Data License:** See [MovieLens Terms of Use](https://grouplens.org/datasets/movielens/)
- **Citation:**
  > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872



For questions or contributions, please open an issue or pull request. 
