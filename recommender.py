"""
Load the precomputed artifacts and serve top-K recommendations from the three
thesis models. No training happens here; everything is read from artifacts/
produced by precompute.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"

MODELS = ["MF", "CatBoost", "LightGCN"]
MODEL_LABELS = {
    "MF": "Matrix Factorisation",
    "CatBoost": "CatBoost (extended, 42 feat)",
    "LightGCN": "LightGCN (feature-augmented)",
}
# short names for tight column headers
MODEL_SHORT = {
    "MF": "Matrix Factorisation",
    "CatBoost": "CatBoost",
    "LightGCN": "LightGCN",
}
MODEL_TAGLINE = {
    "MF": "Collaborative filtering, ratings only",
    "CatBoost": "Gradient boosting, 42 mixed features",
    "LightGCN": "Graph neural network",
}
# models that expose item embeddings usable for similarity search
EMBEDDING_MODELS = ["MF", "LightGCN"]

# ---- shared metadata used by the cold-start "build a taste" page ----
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
AGE_OPTIONS = {  # label -> MovieLens age code
    "Under 18": 1, "18-24": 18, "25-34": 25, "35-44": 35,
    "45-49": 45, "50-55": 50, "56+": 56,
}
OCC_OPTIONS = {  # label -> MovieLens occupation code
    "other / not specified": 0, "academic/educator": 1, "artist": 2,
    "clerical/admin": 3, "college/grad student": 4, "customer service": 5,
    "doctor/health care": 6, "executive/managerial": 7, "farmer": 8,
    "homemaker": 9, "K-12 student": 10, "lawyer": 11, "programmer": 12,
    "retired": 13, "sales/marketing": 14, "scientist": 15, "self-employed": 16,
    "technician/engineer": 17, "tradesman/craftsman": 18, "unemployed": 19,
    "writer": 20,
}
AGE_MIN, AGE_MAX = 1, 56  # MinMaxScaler bounds used at training time


def artifacts_exist():
    needed = ["movies.parquet", "users.parquet", "history.npz", "mf.npz",
              "lgcn.npz", "catboost_ext.cbm", "cb_cache.npz", "metrics.json"]
    return all((ART / f).exists() for f in needed)


class Recommender:
    """Holds every loaded model and exposes scoring and top-K helpers."""

    def __init__(self):
        self.movies = pd.read_parquet(ART / "movies.parquet")
        self.users = pd.read_parquet(ART / "users.parquet")
        with open(ART / "metrics.json") as fh:
            self.metrics = json.load(fh)

        mf = np.load(ART / "mf.npz", allow_pickle=True)
        self.movie_ids = mf["movie_ids"]
        self.user_ids = mf["user_ids"]
        self.user_to_idx = {int(u): i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {int(m): i for i, m in enumerate(self.movie_ids)}

        self._mf = {k: mf[k] for k in ("P", "Q", "b_u", "b_i")}
        self._mf_mu = float(mf["mu"])

        lg = np.load(ART / "lgcn.npz", allow_pickle=True)
        self._lg = {"U": lg["U"], "I": lg["I"], "b_u": lg["b_u"], "b_i": lg["b_i"]}
        self._lg_mu = float(lg["mu"])

        cb = np.load(ART / "cb_cache.npz", allow_pickle=True)
        self._cb_user_block = cb["user_block"]
        self._cb_item_block = cb["item_block"]
        self._occ_pos = int(cb["occ_col_pos"][0])
        self._cb_feature_label = [self._pretty_feature(str(c))
                                  for c in cb["feature_cols"]]
        from catboost import CatBoostRegressor
        self._cb_model = CatBoostRegressor()
        self._cb_model.load_model(str(ART / "catboost_ext.cbm"))

        hist = np.load(ART / "history.npz", allow_pickle=True)
        h_users = hist["user_ids"]
        self._seen = {int(u): set(json.loads(s)) for u, s in zip(h_users, hist["seen"])}
        self._relevant = {int(u): set(json.loads(s)) for u, s in zip(h_users, hist["relevant"])}
        self._train_mean = {int(u): float(m) for u, m in zip(h_users, hist["train_mean"])}

        self._title = dict(zip(self.movies["movie_id"], self.movies["title"]))
        self._genres = dict(zip(self.movies["movie_id"], self.movies["genres"]))

        # popularity (training rating count) for default movie picks and ordering
        pop = {}
        for u, s in self._seen.items():
            for m in s:
                pop[m] = pop.get(m, 0) + 1
        self._popularity = pop
        # movie picker options "Title (year)" -> movie_id, popular first
        order = sorted(self.movie_ids, key=lambda m: -self._popularity.get(int(m), 0))
        self._movie_options = [(self._title.get(int(m), str(m)), int(m)) for m in order]

        # genre one-hot matrix and year array aligned to self.movie_ids order,
        # used for filtering, diversity and cold-start taste construction
        self._genre_mat = np.zeros((len(self.movie_ids), len(ALL_GENRES)), dtype=float)
        gset = {int(r.movie_id): set(str(r.genres).split("|"))
                for r in self.movies.itertuples()}
        year_map = dict(zip(self.movies["movie_id"], self.movies["year"]))
        self._year = np.array([float(year_map.get(int(m), np.nan)) for m in self.movie_ids])
        for row, m in enumerate(self.movie_ids):
            present = gset.get(int(m), set())
            for col, g in enumerate(ALL_GENRES):
                if g in present:
                    self._genre_mat[row, col] = 1.0
        # popularity percentile per movie index (0 = least, 1 = most rated)
        pop_arr = np.array([self._popularity.get(int(m), 0) for m in self.movie_ids],
                           dtype=float)
        ranks = pop_arr.argsort().argsort().astype(float)
        self._pop_pct = ranks / max(1, len(ranks) - 1)

        # lazy caches for live NDCG@k on the held-out test eval set
        self._eval_users_cache = None
        self._eval_rank_cache = {}   # model -> {uid: ranked movie_ids (top MAX_K)}
        self._ndcg_cache = {}        # (model, k) -> float

    # ---- per-model candidate scoring (vectorised over item indices) ----
    def _score_mf(self, uidx, item_idx):
        m = self._mf
        return np.clip(self._mf_mu + m["b_u"][uidx] + m["b_i"][item_idx]
                       + m["P"][uidx] @ m["Q"][item_idx].T, 1.0, 5.0)

    def _score_lgcn(self, uidx, item_idx):
        m = self._lg
        return np.clip(self._lg_mu + m["b_u"][uidx] + m["b_i"][item_idx]
                       + m["U"][uidx] @ m["I"][item_idx].T, 1.0, 5.0)

    def _score_catboost(self, uidx, item_idx):
        urow = self._cb_user_block[uidx]
        irows = self._cb_item_block[item_idx]
        rows = np.column_stack([np.tile(urow, (len(item_idx), 1)), irows]).astype(object)
        rows[:, self._occ_pos] = rows[:, self._occ_pos].astype(int)  # categorical
        return np.clip(self._cb_model.predict(rows), 1.0, 5.0)

    def _score(self, model, uidx, item_idx):
        if model == "MF":
            return self._score_mf(uidx, item_idx)
        if model == "LightGCN":
            return self._score_lgcn(uidx, item_idx)
        return self._score_catboost(uidx, item_idx)

    # ---- public helpers ----
    @staticmethod
    def _pretty_feature(col):
        """Human-readable label for a CatBoost feature name (for explanations)."""
        if col.startswith("ua_"):
            return f"likes {col[3:]} films"
        labels = {
            "gender": "gender", "age_scaled": "age", "occupation": "occupation",
            "year_scaled": "film release year",
            "item_mean_rating": "film's average rating",
            "item_rating_count": "film's number of ratings",
        }
        if col in labels:
            return labels[col]
        return f"is a {col} film"

    def known_user(self, user_id):
        return int(user_id) in self.user_to_idx

    def user_profile(self, user_id):
        row = self.users[self.users["user_id"] == int(user_id)]
        if row.empty:
            return None
        row = row.iloc[0]
        seen = self._seen.get(int(user_id), set())
        return {
            "gender": "Male" if row["gender"] == "M" else "Female",
            "age_band": row["age_band"],
            "occupation": row["occupation_name"],
            "n_rated_train": len(seen),
            "train_mean": self._train_mean.get(int(user_id), float("nan")),
        }

    def liked_history(self, user_id, k=10):
        """A few movies the user rated in the training period (just titles)."""
        seen = list(self._seen.get(int(user_id), set()))[:k]
        return [(m, self._title.get(m, str(m)), self._genres.get(m, "")) for m in seen]

    def recommend(self, user_id, model, k=10):
        user_id = int(user_id)
        uidx = self.user_to_idx.get(user_id)
        if uidx is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])
        seen = self._seen.get(user_id, set())
        cand_mask = np.array([m not in seen for m in self.movie_ids])
        cand_idx = np.where(cand_mask)[0]
        scores = self._score(model, uidx, cand_idx)
        top_local = np.argsort(scores)[::-1][:k]
        top_idx = cand_idx[top_local]
        top_ids = self.movie_ids[top_idx]
        top_scores = scores[top_local]
        return pd.DataFrame({
            "movie_id": top_ids,
            "title": [self._title.get(m, str(m)) for m in top_ids],
            "genres": [self._genres.get(m, "") for m in top_ids],
            "score": np.round(top_scores, 3),
        })

    def recommend_all(self, user_id, k=10):
        return {m: self.recommend(user_id, m, k) for m in MODELS}

    def overlap_matrix(self, user_id, k=10):
        recs = self.recommend_all(user_id, k)
        sets = {m: set(df["movie_id"]) for m, df in recs.items()}
        mat = pd.DataFrame(index=MODELS, columns=MODELS, dtype=int)
        for a in MODELS:
            for b in MODELS:
                mat.loc[a, b] = len(sets[a] & sets[b])
        return mat, recs

    def sample_user_ids(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        return sorted(int(u) for u in rng.choice(self.user_ids, size=n, replace=False))

    # ---- movie helpers ----
    def movie_options(self):
        """List of (label, movie_id) for a picker, most popular first."""
        return self._movie_options

    def movie_title(self, movie_id):
        return self._title.get(int(movie_id), str(movie_id))

    def movie_genres(self, movie_id):
        return self._genres.get(int(movie_id), "")

    def popularity(self, movie_id):
        return self._popularity.get(int(movie_id), 0)

    # ---- live ranking quality (NDCG@k on the held-out test set) ----
    MAX_EVAL_K = 20

    def _eval_users(self, n=200, seed=42):
        """Same evaluation sample used offline: users with >=1 relevant test item."""
        if self._eval_users_cache is None:
            cand = sorted(u for u, rel in self._relevant.items() if len(rel) > 0)
            rng = np.random.default_rng(seed)
            pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
            self._eval_users_cache = [int(u) for u in pick]
        return self._eval_users_cache

    def _eval_rankings(self, model):
        """Top-MAX_EVAL_K recommended movie ids per eval user, scored once and cached."""
        if model not in self._eval_rank_cache:
            ranks = {}
            for uid in self._eval_users():
                uidx = self.user_to_idx[uid]
                seen = self._seen.get(uid, set())
                cand_idx = np.where([m not in seen for m in self.movie_ids])[0]
                scores = self._score(model, uidx, cand_idx)
                top_local = np.argsort(scores)[::-1][:self.MAX_EVAL_K]
                ranks[uid] = self.movie_ids[cand_idx[top_local]]
            self._eval_rank_cache[model] = ranks
        return self._eval_rank_cache[model]

    def headline_metric(self, model, k=10):
        """Mean NDCG@k over the held-out test eval set (same protocol as offline).

        k == 10 reuses the cached benchmark in metrics.json; other k are computed
        live from a single per-user ranking and memoised, so moving the slider is
        cheap after the first time a model is evaluated.
        """
        if k == 10 and model in self.metrics and "ndcg10" in self.metrics[model]:
            return self.metrics[model]["ndcg10"]
        if (model, k) in self._ndcg_cache:
            return self._ndcg_cache[(model, k)]
        k = min(int(k), self.MAX_EVAL_K)
        rankings = self._eval_rankings(model)
        nd = []
        for uid in self._eval_users():
            relevant = self._relevant[uid]
            hits = [1.0 if m in relevant else 0.0 for m in rankings[uid][:k]]
            dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
            nd.append(dcg / ideal if ideal > 0 else 0.0)
        value = float(np.mean(nd))
        self._ndcg_cache[(model, k)] = value
        return value

    def similar_movies(self, movie_id, model="LightGCN", k=10):
        """Nearest items in a model's embedding space by cosine similarity."""
        idx = self.movie_to_idx.get(int(movie_id))
        if idx is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "similarity"])
        E = self._lg["I"] if model == "LightGCN" else self._mf["Q"]
        v = E[idx]
        denom = np.linalg.norm(E, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9
        sims = (E @ v) / denom
        order = np.argsort(sims)[::-1]
        order = [j for j in order if j != idx][:k]
        ids = self.movie_ids[order]
        return pd.DataFrame({
            "movie_id": ids,
            "title": [self._title.get(int(m), str(m)) for m in ids],
            "genres": [self._genres.get(int(m), "") for m in ids],
            "similarity": np.round(sims[order], 3),
        })

    # ---- filtering helpers (genre / decade constraints on candidates) ----
    def genres_list(self):
        return list(ALL_GENRES)

    def year_bounds(self):
        yrs = self._year[~np.isnan(self._year)]
        return int(yrs.min()), int(yrs.max())

    def _candidate_mask(self, seen, genres=None, min_year=None, max_year=None):
        """Boolean mask over self.movie_ids of allowed candidate items."""
        mask = np.array([int(m) not in seen for m in self.movie_ids])
        if genres:
            cols = [ALL_GENRES.index(g) for g in genres if g in ALL_GENRES]
            if cols:
                mask &= (self._genre_mat[:, cols].sum(axis=1) > 0)
        if min_year is not None:
            mask &= np.nan_to_num(self._year, nan=-1) >= min_year
        if max_year is not None:
            mask &= np.nan_to_num(self._year, nan=10 ** 9) <= max_year
        return mask

    def _df_from_idx(self, idx, scores, score_name="score", round_to=3):
        ids = self.movie_ids[idx]
        return pd.DataFrame({
            "movie_id": ids,
            "title": [self._title.get(int(m), str(m)) for m in ids],
            "genres": [self._genres.get(int(m), "") for m in ids],
            score_name: np.round(scores, round_to),
        })

    def recommend_filtered(self, user_id, model, k=10,
                           genres=None, min_year=None, max_year=None):
        """Top-K for an existing user, optionally constrained by genre/decade."""
        uidx = self.user_to_idx.get(int(user_id))
        if uidx is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])
        seen = self._seen.get(int(user_id), set())
        cand_idx = np.where(self._candidate_mask(seen, genres, min_year, max_year))[0]
        if len(cand_idx) == 0:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])
        scores = self._score(model, uidx, cand_idx)
        top_local = np.argsort(scores)[::-1][:k]
        top_idx = cand_idx[top_local]
        return self._df_from_idx(top_idx, scores[top_local])

    # ---- cold-start: recommend for a brand-new, unseen user ----
    def _foldin_vector(self, E, b_i, mu, liked_idx, target=5.0, reg=0.15):
        """Ridge fold-in: estimate a latent user vector from liked item factors.

        Solves (Q'Q + reg.I) p = Q'(target - mu - b_i) over the liked items, the
        standard way to place a cold user in an existing factor space.
        """
        Ql = E[liked_idx]
        y = target - mu - b_i[liked_idx]
        d = Ql.shape[1]
        A = Ql.T @ Ql + reg * np.eye(d)
        return np.linalg.solve(A, Ql.T @ y)

    def _cold_user_row_cb(self, gender, age_code, occupation_code, liked_idx):
        """Build the 21-feature CatBoost user block for a synthetic user."""
        gender_val = 1.0 if gender == "Male" else 0.0
        age_scaled = (age_code - AGE_MIN) / (AGE_MAX - AGE_MIN)
        # genre affinity = how often the liked films carry each genre
        genre_ind = self._cb_item_block[liked_idx, 1:1 + len(ALL_GENRES)]
        ua = genre_ind.mean(axis=0) if len(liked_idx) else np.zeros(len(ALL_GENRES))
        return np.concatenate([[gender_val, age_scaled, float(occupation_code)], ua])

    def cold_start_recommend(self, liked_movie_ids, gender, age_code,
                             occupation_code, k=10, genres=None,
                             min_year=None, max_year=None):
        """Live recommendations for an unseen user described by a few liked films
        plus demographics. Returns {model: DataFrame}."""
        liked_idx = np.array([self.movie_to_idx[int(m)] for m in liked_movie_ids
                              if int(m) in self.movie_to_idx], dtype=int)
        seen = {int(m) for m in liked_movie_ids}
        cand_idx = np.where(self._candidate_mask(seen, genres, min_year, max_year))[0]
        out = {}
        if len(liked_idx) == 0 or len(cand_idx) == 0:
            empty = pd.DataFrame(columns=["movie_id", "title", "genres", "score"])
            return {m: empty for m in MODELS}

        # MF and LightGCN: fold-in user vector, then score candidates
        for model in EMBEDDING_MODELS:
            if model == "MF":
                E, b_i, mu = self._mf["Q"], self._mf["b_i"], self._mf_mu
            else:
                E, b_i, mu = self._lg["I"], self._lg["b_i"], self._lg_mu
            p = self._foldin_vector(E, b_i, mu, liked_idx)
            scores = np.clip(mu + b_i[cand_idx] + E[cand_idx] @ p, 1.0, 5.0)
            top_local = np.argsort(scores)[::-1][:k]
            out[model] = self._df_from_idx(cand_idx[top_local], scores[top_local])

        # CatBoost: construct the synthetic user row and score candidates
        urow = self._cold_user_row_cb(gender, age_code, occupation_code, liked_idx)
        irows = self._cb_item_block[cand_idx]
        rows = np.column_stack([np.tile(urow, (len(cand_idx), 1)), irows]).astype(object)
        rows[:, self._occ_pos] = rows[:, self._occ_pos].astype(int)
        cb_scores = np.clip(self._cb_model.predict(rows), 1.0, 5.0)
        top_local = np.argsort(cb_scores)[::-1][:k]
        out["CatBoost"] = self._df_from_idx(cand_idx[top_local], cb_scores[top_local])
        return {m: out[m] for m in MODELS}

    # ---- live ranking metric curve and catalogue diagnostics ----
    def metrics_at_k(self, model, k):
        """(precision@k, recall@k, ndcg@k) on the held-out eval set."""
        k = min(int(k), self.MAX_EVAL_K)
        rankings = self._eval_rankings(model)
        p, r, nd = [], [], []
        for uid in self._eval_users():
            relevant = self._relevant[uid]
            hits = [1.0 if m in relevant else 0.0 for m in rankings[uid][:k]]
            p.append(sum(hits) / k)
            r.append(sum(hits) / len(relevant) if relevant else 0.0)
            dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
            nd.append(dcg / ideal if ideal > 0 else 0.0)
        return float(np.mean(p)), float(np.mean(r)), float(np.mean(nd))

    def metric_curve(self, ks=range(1, 21)):
        """DataFrame of NDCG@k per model across a range of K for line charts."""
        data = {}
        for model in MODELS:
            data[MODEL_SHORT[model]] = [self.metrics_at_k(model, k)[2] for k in ks]
        return pd.DataFrame(data, index=list(ks))

    def catalogue_stats(self, k=10):
        """Per-model coverage, average popularity and list diversity over the
        eval set — reveals popularity bias and how varied each list is."""
        rows = []
        norm = self._genre_mat / (np.linalg.norm(self._genre_mat, axis=1,
                                                  keepdims=True) + 1e-9)
        for model in MODELS:
            rankings = self._eval_rankings(model)
            recommended, pops, divs = set(), [], []
            for uid in self._eval_users():
                idx = [self.movie_to_idx[int(m)] for m in rankings[uid][:k]]
                recommended.update(idx)
                pops.extend(self._pop_pct[idx])
                if len(idx) > 1:
                    G = norm[idx]
                    sim = G @ G.T
                    n = len(idx)
                    off = (sim.sum() - np.trace(sim)) / (n * (n - 1))
                    divs.append(1.0 - off)
            rows.append({
                "Model": MODEL_SHORT[model],
                "Catalogue coverage": len(recommended) / len(self.movie_ids),
                "Avg popularity": float(np.mean(pops)) if pops else 0.0,
                "List diversity": float(np.mean(divs)) if divs else 0.0,
            })
        return pd.DataFrame(rows).set_index("Model")

    # ---- per-recommendation explanations ----
    def explain_embedding(self, liked_movie_ids, movie_id, model="LightGCN", top=2):
        """Which of the user's liked films most resemble a recommended title,
        in the chosen model's embedding space. Returns a list of titles."""
        E = self._lg["I"] if model == "LightGCN" else self._mf["Q"]
        tgt = self.movie_to_idx.get(int(movie_id))
        if tgt is None:
            return []
        liked_idx = [self.movie_to_idx[int(m)] for m in liked_movie_ids
                     if int(m) in self.movie_to_idx and int(m) != int(movie_id)]
        if not liked_idx:
            return []
        v = E[tgt]
        L = E[liked_idx]
        sims = (L @ v) / (np.linalg.norm(L, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9)
        best = np.argsort(sims)[::-1][:top]
        return [self._title.get(int(self.movie_ids[liked_idx[b]]), "") for b in best]

    def explain_catboost(self, user_row_or_id, movie_id, top=3):
        """Top SHAP feature contributions for one CatBoost prediction.

        user_row_or_id may be an existing user id (int) or a prebuilt 21-vector
        for a cold user. Returns list of (feature_label, signed_contribution)."""
        midx = self.movie_to_idx.get(int(movie_id))
        if midx is None:
            return []
        if isinstance(user_row_or_id, (int, np.integer)):
            uidx = self.user_to_idx.get(int(user_row_or_id))
            if uidx is None:
                return []
            urow = self._cb_user_block[uidx]
        else:
            urow = np.asarray(user_row_or_id, dtype=float)
        row = np.concatenate([urow, self._cb_item_block[midx]]).astype(object)[None, :]
        row[:, self._occ_pos] = row[:, self._occ_pos].astype(int)
        from catboost import Pool
        shap = self._cb_model.get_feature_importance(
            Pool(row, cat_features=[self._occ_pos]), type="ShapValues")[0]
        contribs = shap[:-1]  # last entry is the expected-value bias
        names = self._cb_feature_label
        order = np.argsort(np.abs(contribs))[::-1][:top]
        return [(names[i], float(contribs[i])) for i in order]


