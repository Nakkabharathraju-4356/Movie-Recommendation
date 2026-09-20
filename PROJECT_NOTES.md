# Recruiter Presentation Notes

## What makes this different from a tutorial

The repository should tell a clear engineering story:

**Problem → Data leakage control → Baseline → Multiple retrieval strategies → Ranking evaluation → Bias/coverage → Cold start → Explainability → Application**

Do not describe it as simply "a movie recommender using cosine similarity."

## Suggested GitHub repository name

`movie-recommender-benchmark`

## Suggested repository description

> Production-style movie recommendation system benchmarking collaborative, content-based and hybrid models with time-aware evaluation, ranking metrics, cold-start handling and a Streamlit app.

## Recommended screenshots for GitHub

After running the application, capture:

1. Recommendation screen
2. Model benchmark table
3. Recent rating history
4. VS Code project structure

Put them in `docs/screenshots/` if you want to enhance the README.

## Interview architecture answer

"I treated recommendation as a ranking problem rather than only a rating-prediction problem. I used a time-based split, established popularity as a baseline, compared item-based collaborative filtering and latent factors, added content similarity for cold-start and hybrid retrieval, and evaluated with ranking metrics plus coverage. I then exposed the trained models through a Streamlit application."

That explanation demonstrates understanding of the complete lifecycle rather than only model training.
