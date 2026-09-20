from models.popularity import PopularityRecommender
from models.item_knn import ItemKNNRecommender
from models.svd import SVDRecommender
from models.content import ContentBasedRecommender
from models.hybrid import HybridRecommender


def build_models(train, movies):
    popularity = PopularityRecommender(min_ratings=20).fit(train)
    item_knn = ItemKNNRecommender(n_neighbors=40).fit(train)
    svd = SVDRecommender(n_components=40).fit(train)
    content = ContentBasedRecommender().fit(movies)

    hybrid = HybridRecommender(
        collaborative=svd,
        content=content,
        popularity=popularity
    )

    return {
        "Popularity": popularity,
        "Item-kNN": item_knn,
        "SVD": svd,
        "Content-Based": content,
        "Hybrid": hybrid,
    }
