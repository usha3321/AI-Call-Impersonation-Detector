from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarity(features_1, features_2):
    features_1 = np.array(features_1).reshape(1, -1)
    features_2 = np.array(features_2).reshape(1, -1)

    similarity_score = cosine_similarity(
        features_1,
        features_2
    )

    return similarity_score[0][0]