import pandas as pd
import numpy  as np
from   sklearn.metrics.pairwise import cosine_similarity

class GenresBasedFilter(object):
    def __init__(self, dataframe, k=5):
        self.artist_to_idx = {row["artistID"]: idx for idx, row in dataframe.iterrows()}
        self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
        self.k = k

        tagID = set(g for G in dataframe['tagID'] for g in G)
        for g in tagID:
            dataframe[g] = dataframe.tagID.transform(lambda x: int(g in x))

        self.artist_genres = dataframe.drop(columns=['artistID', 'name', 'tagID'])

    def fit(self, ratings):
        self.dataframe_cosine_sim_ = cosine_similarity(self.artist_genres, self.artist_genres)

        self.user_ratings_ = {}
        for (user_id, artist_id, rating) in ratings.build_testset():
            if user_id not in self.user_ratings_:
                self.user_ratings_[user_id] = {}
            self.user_ratings_[user_id][artist_id] = rating

        return self

    def predict(self, user, artist):
        if not user in self.user_ratings_ or not artist in self.artist_to_idx:
            global_mean = np.mean([
                rating for dataframe in self.user_ratings_.values() for rating in dataframe.values()
            ])
            return global_mean

        artist_idx = self.artist_to_idx[artist]
        sim_scores = list(enumerate(self.dataframe_cosine_sim_[artist_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]

        sims = []

        for artist, score in sim_scores:
            if self.idx_to_artist[artist] in self.user_ratings_[user]:
                sims.append((self.user_ratings_[user][self.idx_to_artist[artist]], score))
                if len(sims) >= self.k:
                    break

        user_mean = np.mean(list(self.user_ratings_[user].values()))

        pred = 0
        sim_sum = 0

        for rating, score in sims:
            pred += score * (rating - user_mean)
            sim_sum += score

        if sim_sum == 0:
            return user_mean

        return user_mean + pred / sim_sum
