# Movie Recommendation System — Professional Benchmarking Project

A production-style recommendation system that benchmarks **Popularity, Item-kNN Collaborative Filtering, SVD/Latent Factors, Content-Based Filtering, and a Hybrid Recommender** using a **time-based evaluation split** and ranking metrics.

## Why this project is resume-worthy

This project is intentionally built beyond a basic cosine-similarity notebook:

- Time-aware train/test split to reduce temporal leakage
- Multiple recommendation algorithms
- Popularity baseline
- Precision@K, Recall@K, NDCG@K, MAP@K
- Catalog coverage and recommendation diversity
- Cold-start fallback
- Popularity-bias analysis
- Explainable recommendations
- Streamlit interactive application
- Automated tests
- Reproducible training/evaluation pipeline
- Clean separation of data, models, metrics, and UI

## Architecture

```text
movie-recommender/
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   ├── popularity.py
│   ├── item_knn.py
│   ├── svd.py
│   ├── content.py
│   └── hybrid.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── utils.py
├── tests/
│   ├── test_metrics.py
│   └── test_recommenders.py
├── artifacts/
├── requirements.txt
├── train.py
├── evaluate.py
├── download_data.py
└── README.md
```

## 1. Requirements

- Python 3.10+
- VS Code
- Internet connection for downloading MovieLens data

## 2. Setup in VS Code

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Download MovieLens

The default project uses MovieLens 1M.

```bash
python download_data.py
```

The script downloads the public MovieLens 1M dataset and converts it into:

```text
data/ratings.csv
data/movies.csv
```

## 4. Train the benchmark

```bash
python train.py
```

This creates reusable artifacts in `artifacts/`.

## 5. Run evaluation

```bash
python evaluate.py
```

A benchmark table is saved to:

```text
artifacts/benchmark_results.csv
```

## 6. Launch the application

```bash
streamlit run app/streamlit_app.py
```

Open the local URL shown by Streamlit.

## 7. Run tests

```bash
pytest -q
```

## Methodology

### Time-based split

For every user, interactions are ordered by timestamp. The most recent interactions are held out for evaluation.

This better resembles production recommendation, where the model learns from the past and recommends for future interactions.

### Models

1. **Popularity**
   - Strong baseline
   - Uses Bayesian-style weighted rating and popularity
2. **Item-kNN**
   - Collaborative filtering based on item-item similarity
3. **SVD**
   - Latent-factor collaborative filtering
4. **Content-Based**
   - TF-IDF over movie genres/title
5. **Hybrid**
   - Blends collaborative, content and popularity signals

### Metrics

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- Catalog Coverage
- Intra-list Diversity

## Cold start

For a new user with no interaction history, the application falls back to a popularity-based recommendation.

For a new/unknown movie, the content representation can be used when metadata is available.

## Explainability

The UI provides explanations such as:

- "Recommended because you liked movies similar to..."
- "Popular among users with similar tastes"
- "Matches genres you frequently rate highly"

These explanations are intentionally based on available model signals rather than fabricated claims.

## Resume bullets

- Built a production-style movie recommendation system benchmarking popularity, item-kNN, latent-factor SVD, content-based TF-IDF and hybrid recommendation approaches.
- Designed a time-based evaluation framework with Precision@K, Recall@K, NDCG@K, MAP@K, catalog coverage and diversity metrics.
- Implemented cold-start fallbacks, popularity-bias analysis, explainable recommendations and an interactive Streamlit application.
- Automated model training, evaluation and regression tests using a modular Python architecture.

## Interview discussion points

Be prepared to explain:

- Why time-based splitting is preferred to a random split
- Why RMSE alone is insufficient for recommendation ranking
- How cold-start users are handled
- Why popularity is a necessary baseline
- Precision vs Recall vs NDCG
- Why diversity and coverage matter
- How the system could be A/B tested
- How the architecture could be moved to production
- How embeddings/vector databases could improve retrieval at larger scale

## Production roadmap

A real production system could add:

- Candidate generation + ranking architecture
- ANN/vector retrieval
- Feature store
- Online feedback loops
- A/B testing
- Model registry
- Batch/streaming pipelines
- Redis caching
- FastAPI inference service
- Docker + CI/CD
- Monitoring for drift, coverage and popularity concentration

## Dataset

MovieLens 1M is provided by GroupLens Research at the University of Minnesota. The project downloads it from the public GroupLens distribution when `download_data.py` is run. Review the dataset's own usage/license terms before redistribution.
