"""
Train the three core thesis models once and save lightweight artifacts the
demo website can load instantly. Run this from the RSProj environment:

    conda activate RSProj
    python precompute.py

It reproduces the pipeline of RS_Code_Implementation.ipynb (time-based split,
shared mixed-type features) for Matrix Factorisation, CatBoost (extended, 42
features) and feature-augmented LightGCN, then writes:

    artifacts/movies.parquet      item metadata (title, year, genres)
    artifacts/users.parquet       user metadata (gender, age band, occupation)
    artifacts/history.npz         per-user train history and test relevant sets
    artifacts/mf.npz              MF factors and biases
    artifacts/lgcn.npz            LightGCN embeddings and biases
    artifacts/catboost_ext.cbm    trained CatBoost model
    artifacts/cb_cache.npz        cached user/item feature blocks for CatBoost
    artifacts/metrics.json        RMSE/MAE and ranking metrics per model
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "datasets"
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

np.random.seed(42)

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
              45: "45-49", 50: "50-55", 56: "56+"}
OCC_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}


def load_raw():
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat", sep="::", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"], engine="python",
    )
    users = pd.read_csv(
        DATA_DIR / "users.dat", sep="::", header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"], engine="python",
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat", sep="::", header=None,
        names=["movie_id", "title", "genres"], encoding="latin-1", engine="python",
    )
    return ratings, users, movies


def build_features(users, movies):
    users_feat = users[["user_id", "gender", "age", "occupation"]].copy()
    users_feat["gender"] = (users_feat["gender"] == "M").astype(int)
    users_feat["age_scaled"] = MinMaxScaler().fit_transform(users_feat[["age"]])

    movies_feat = movies[["movie_id", "title", "genres"]].copy()
    movies_feat["year"] = movies_feat["title"].str.extract(r"\((\d{4})\)").astype(float)
    for g in ALL_GENRES:
        movies_feat[g] = movies_feat["genres"].str.contains(g, regex=False).astype(int)
    movies_feat["year_scaled"] = MinMaxScaler().fit_transform(movies_feat[["year"]])
    return users_feat, movies_feat


def time_split(ratings):
    rs = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(rs)
    train = rs.iloc[: int(n * 0.80)].copy()
    val = rs.iloc[int(n * 0.80): int(n * 0.90)].copy()
    test = rs.iloc[int(n * 0.90):].copy()
    return train, val, test


# Matrix Factorisation (SGD with biases), matching the notebook recipe
class MatrixFactorisation:
    def __init__(self, n_users, n_items, user_to_idx, movie_to_idx,
                 k=20, lr=0.005, reg=0.02, n_epochs=20):
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
        self.k, self.lr, self.reg, self.n_epochs = k, lr, reg, n_epochs
        self.P = np.random.normal(0, 0.1, (n_users, k))
        self.Q = np.random.normal(0, 0.1, (n_items, k))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.mu = 0.0

    def fit(self, train_df, val_df=None):
        self.mu = train_df["rating"].mean()
        u_idx = train_df["user_id"].map(self.user_to_idx).values
        i_idx = train_df["movie_id"].map(self.movie_to_idx).values
        r = train_df["rating"].values.astype(float)
        for epoch in range(self.n_epochs):
            order = np.random.permutation(len(r))
            for idx in order:
                u, i, rui = u_idx[idx], i_idx[idx], r[idx]
                e = rui - (self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i])
                self.P[u] += self.lr * (e * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (e * self.P[u] - self.reg * self.Q[i])
                self.b_u[u] += self.lr * (e - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (e - self.reg * self.b_i[i])
            if val_df is not None and (epoch + 1) % 5 == 0:
                print(f"  MF epoch {epoch + 1:>2}/{self.n_epochs}  val RMSE: {self._rmse(val_df):.4f}")

    def predict_pair(self, u, i):
        return float(np.clip(self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i], 1.0, 5.0))

    def _rmse(self, df):
        u = df["user_id"].map(self.user_to_idx).values
        i = df["movie_id"].map(self.movie_to_idx).values
        pred = np.clip(self.mu + self.b_u[u] + self.b_i[i]
                       + (self.P[u] * self.Q[i]).sum(1), 1.0, 5.0)
        return float(np.sqrt(mean_squared_error(df["rating"].values, pred)))


def ranking_metrics(score_user_fn, test_df, train_df, movie_id_arr,
                    K=10, threshold=4.0, n_users=200, seed=42):
    train_seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_relevant = (
        test_df[test_df["rating"] >= threshold]
        .groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    rng = np.random.default_rng(seed)
    candidates = [u for u in test_relevant if len(test_relevant[u]) > 0]
    sampled = rng.choice(candidates, size=min(n_users, len(candidates)), replace=False)
    p, r, nd = [], [], []
    for uid in sampled:
        seen = train_seen.get(uid, set())
        unrated_mask = np.array([m not in seen for m in movie_id_arr])
        unrated_idx = np.where(unrated_mask)[0]
        scores = score_user_fn(uid, unrated_idx)
        top_local = np.argsort(scores)[::-1][:K]
        top_movies = movie_id_arr[unrated_idx[top_local]]
        relevant = test_relevant[uid]
        hits = [1.0 if m in relevant else 0.0 for m in top_movies]
        p.append(sum(hits) / K)
        r.append(sum(hits) / len(relevant) if relevant else 0.0)
        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(K, len(relevant))))
        nd.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(p)), float(np.mean(r)), float(np.mean(nd))


def main():
    print("Loading MovieLens-1M...")
    ratings, users, movies = load_raw()
    users_feat, movies_feat = build_features(users, movies)
    train_raw, val_raw, test_raw = time_split(ratings)

    all_user_ids = sorted(ratings["user_id"].unique())
    all_movie_ids = sorted(ratings["movie_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(all_user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(all_movie_ids)}
    movie_id_arr = np.array(all_movie_ids)
    n_u, n_m = len(all_user_ids), len(all_movie_ids)
    print(f"  users={n_u:,}  movies={n_m:,}  ratings={len(ratings):,}")

    metrics = {}

    # ---- Matrix Factorisation ----
    print("Training Matrix Factorisation...")
    mf = MatrixFactorisation(n_u, n_m, user_to_idx, movie_to_idx)
    if (ART / "mf.npz").exists():
        print("  reusing saved mf.npz")
        d = np.load(ART / "mf.npz", allow_pickle=True)
        mf.P, mf.Q = d["P"], d["Q"]
        mf.b_u, mf.b_i, mf.mu = d["b_u"], d["b_i"], float(d["mu"])
    else:
        mf.fit(train_raw, val_df=val_raw)
        np.savez(
            ART / "mf.npz", P=mf.P, Q=mf.Q, b_u=mf.b_u, b_i=mf.b_i, mu=mf.mu,
            user_ids=np.array(all_user_ids), movie_ids=movie_id_arr,
        )
    u = test_raw["user_id"].map(user_to_idx).values
    i = test_raw["movie_id"].map(movie_to_idx).values
    mf_pred = np.clip(mf.mu + mf.b_u[u] + mf.b_i[i] + (mf.P[u] * mf.Q[i]).sum(1), 1.0, 5.0)
    metrics["MF"] = {
        "rmse": float(np.sqrt(mean_squared_error(test_raw["rating"], mf_pred))),
        "mae": float(mean_absolute_error(test_raw["rating"], mf_pred)),
    }

    def mf_score_user(uid, item_idx):
        uu = user_to_idx.get(uid)
        if uu is None:
            return np.full(len(item_idx), mf.mu)
        return np.clip(mf.mu + mf.b_u[uu] + mf.b_i[item_idx]
                       + mf.P[uu] @ mf.Q[item_idx].T, 1.0, 5.0)

    # ---- CatBoost (extended, 42 features) ----
    print("Training CatBoost (extended)...")
    from catboost import CatBoostRegressor, Pool

    # user genre affinity from training ratings only (no leakage)
    train_join = train_raw.merge(movies_feat[["movie_id"] + ALL_GENRES], on="movie_id")
    ua = train_join.groupby("user_id")[ALL_GENRES].mean()
    ua.columns = [f"ua_{g}" for g in ALL_GENRES]
    ua = ua.reindex(all_user_ids).fillna(0.0)

    item_stats = train_raw.groupby("movie_id")["rating"].agg(["mean", "count"])
    item_stats.columns = ["item_mean_rating", "item_rating_count"]
    item_stats = item_stats.reindex(all_movie_ids)
    item_stats["item_mean_rating"] = item_stats["item_mean_rating"].fillna(train_raw["rating"].mean())
    item_stats["item_rating_count"] = item_stats["item_rating_count"].fillna(0.0)

    UA_COLS = [f"ua_{g}" for g in ALL_GENRES]
    FEATURE_COLS = (
        ["gender", "age_scaled", "occupation"] + UA_COLS
        + ["year_scaled"] + ALL_GENRES + ["item_mean_rating", "item_rating_count"]
    )
    CAT_COLS = ["occupation"]

    users_ext = users_feat.merge(ua, left_on="user_id", right_index=True)
    movies_ext = movies_feat.merge(item_stats, left_on="movie_id", right_index=True)

    def build_cb_df(rdf):
        df = rdf.merge(
            users_ext[["user_id", "gender", "age_scaled", "occupation"] + UA_COLS],
            on="user_id")
        df = df.merge(
            movies_ext[["movie_id", "year_scaled"] + ALL_GENRES
                       + ["item_mean_rating", "item_rating_count"]],
            on="movie_id")
        return df

    train_cb = build_cb_df(train_raw)
    val_cb = build_cb_df(val_raw)
    test_cb = build_cb_df(test_raw)

    cb_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE",
        eval_metric="RMSE", random_seed=42, early_stopping_rounds=20, verbose=100,
    )
    if (ART / "catboost_ext.cbm").exists():
        print("  reusing saved catboost_ext.cbm")
        cb_model.load_model(str(ART / "catboost_ext.cbm"))
    else:
        cb_model.fit(
            Pool(train_cb[FEATURE_COLS], train_cb["rating"], cat_features=CAT_COLS),
            eval_set=Pool(val_cb[FEATURE_COLS], val_cb["rating"], cat_features=CAT_COLS),
        )
        cb_model.save_model(str(ART / "catboost_ext.cbm"))
    cb_pred = np.clip(cb_model.predict(test_cb[FEATURE_COLS]), 1.0, 5.0)
    metrics["CatBoost"] = {
        "rmse": float(np.sqrt(mean_squared_error(test_cb["rating"], cb_pred))),
        "mae": float(mean_absolute_error(test_cb["rating"], cb_pred)),
    }

    # feature blocks for fast candidate scoring at serve time
    cb_user_block = users_ext.set_index("user_id")[
        ["gender", "age_scaled", "occupation"] + UA_COLS].reindex(all_user_ids)
    cb_item_block = movies_ext.set_index("movie_id")[
        ["year_scaled"] + ALL_GENRES + ["item_mean_rating", "item_rating_count"]
    ].reindex(all_movie_ids)
    np.savez(
        ART / "cb_cache.npz",
        user_ids=np.array(all_user_ids),
        movie_ids=movie_id_arr,
        user_block=cb_user_block.values.astype(np.float64),
        item_block=cb_item_block.values.astype(np.float64),
        feature_cols=np.array(FEATURE_COLS),
        occ_col_pos=np.array([2]),  # index of 'occupation' inside the user block
    )

    def cb_score_user(uid, item_idx):
        uu = user_to_idx.get(uid)
        if uu is None:
            return np.full(len(item_idx), train_raw["rating"].mean())
        urow = cb_user_block.values[uu]
        irows = cb_item_block.values[item_idx]
        rows = np.column_stack([np.tile(urow, (len(item_idx), 1)), irows]).astype(object)
        rows[:, 2] = rows[:, 2].astype(int)  # occupation is categorical, not float
        return np.clip(cb_model.predict(rows), 1.0, 5.0)

    # ---- LightGCN (feature-augmented) ----
    print("Training LightGCN...")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn.conv import LGConv

    users_fs = users_feat.set_index("user_id").reindex(all_user_ids)
    movies_fs = movies_feat.set_index("movie_id").reindex(all_movie_ids)
    occ_onehot = (
        pd.get_dummies(users_fs["occupation"].astype(int), dtype=np.float32)
        .reindex(columns=list(range(21)), fill_value=0.0).values
    )
    user_feat_mat = np.hstack([
        users_fs[["gender", "age_scaled"]].values.astype(np.float32), occ_onehot])
    item_feat_mat = movies_fs[["year_scaled"] + ALL_GENRES].values.astype(np.float32)
    USER_FEAT_DIM, ITEM_FEAT_DIM = user_feat_mat.shape[1], item_feat_mat.shape[1]
    EMB_DIM, N_LAYERS = 64, 2

    user_feats_t = torch.tensor(user_feat_mat)
    item_feats_t = torch.tensor(item_feat_mat)
    item_offset = n_u

    tr_u = torch.tensor(train_raw["user_id"].map(user_to_idx).values, dtype=torch.long)
    tr_i = torch.tensor(train_raw["movie_id"].map(movie_to_idx).values, dtype=torch.long)
    tr_r = torch.tensor(train_raw["rating"].values, dtype=torch.float32)
    va_u = torch.tensor(val_raw["user_id"].map(user_to_idx).values, dtype=torch.long)
    va_i = torch.tensor(val_raw["movie_id"].map(movie_to_idx).values, dtype=torch.long)
    va_r = torch.tensor(val_raw["rating"].values, dtype=torch.float32)

    src = torch.cat([tr_u, tr_i + item_offset])
    dst = torch.cat([tr_i + item_offset, tr_u])
    edge_index = torch.stack([src, dst], dim=0)
    mu_train = float(train_raw["rating"].mean())

    class FeatureInitLightGCN(nn.Module):
        def __init__(self, n_users, n_items, ufd, ifd, emb_dim, n_layers, gm):
            super().__init__()
            self.n_users, self.mu = n_users, gm
            self.user_emb = nn.Embedding(n_users, emb_dim)
            self.item_emb = nn.Embedding(n_items, emb_dim)
            self.user_proj = nn.Linear(ufd, emb_dim)
            self.item_proj = nn.Linear(ifd, emb_dim)
            self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
            self.user_bias = nn.Parameter(torch.zeros(n_users))
            self.item_bias = nn.Parameter(torch.zeros(n_items))
            nn.init.normal_(self.user_emb.weight, std=0.1)
            nn.init.normal_(self.item_emb.weight, std=0.1)

        def forward(self, uf, itf, ei):
            x = torch.cat([self.user_emb.weight + self.user_proj(uf),
                           self.item_emb.weight + self.item_proj(itf)], dim=0)
            all_embs = [x]
            for conv in self.convs:
                x = conv(x, ei)
                all_embs.append(x)
            final = torch.stack(all_embs, dim=0).mean(dim=0)
            return final[:self.n_users], final[self.n_users:]

        def predict(self, ui, ii, ue, ie):
            return self.mu + self.user_bias[ui] + self.item_bias[ii] + (ue[ui] * ie[ii]).sum(-1)

    torch.manual_seed(42)
    model = FeatureInitLightGCN(n_u, n_m, USER_FEAT_DIM, ITEM_FEAT_DIM, EMB_DIM, N_LAYERS, mu_train)
    if (ART / "lgcn.npz").exists():
        print("  reusing saved lgcn.npz")
        d = np.load(ART / "lgcn.npz", allow_pickle=True)
        U, I = d["U"], d["I"]
        bu, bi = d["b_u"], d["b_i"]
    else:
        optimizer = torch.optim.Adam([
            {"params": [model.user_bias, model.item_bias], "lr": 0.05},
            {"params": list(model.user_emb.parameters()) + list(model.item_emb.parameters())
                       + list(model.user_proj.parameters()) + list(model.item_proj.parameters()),
             "lr": 0.01},
        ], weight_decay=1e-5)

        N_EPOCHS, BATCH_SIZE, PATIENCE = 30, 16384, 5
        N_TRAIN = tr_u.shape[0]
        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            perm = torch.randperm(N_TRAIN)
            for start in range(0, N_TRAIN, BATCH_SIZE):
                b = perm[start:start + BATCH_SIZE]
                optimizer.zero_grad()
                ue, ie = model(user_feats_t, item_feats_t, edge_index)
                loss = F.mse_loss(model.predict(tr_u[b], tr_i[b], ue, ie), tr_r[b])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                ue, ie = model(user_feats_t, item_feats_t, edge_index)
                vp = model.predict(va_u, va_i, ue, ie).clamp(1.0, 5.0)
                vr = torch.sqrt(F.mse_loss(vp, va_r)).item()
            if vr < best_val - 1e-4:
                best_val, patience = vr, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"  LightGCN early stop at epoch {epoch} (best val RMSE {best_val:.4f})")
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ue, ie = model(user_feats_t, item_feats_t, edge_index)
        U, I = ue.numpy(), ie.numpy()
        bu, bi = model.user_bias.detach().numpy(), model.item_bias.detach().numpy()
        np.savez(ART / "lgcn.npz", U=U, I=I, b_u=bu, b_i=bi, mu=mu_train,
                 user_ids=np.array(all_user_ids), movie_ids=movie_id_arr)

    tu = test_raw["user_id"].map(user_to_idx).values
    ti = test_raw["movie_id"].map(movie_to_idx).values
    lg_pred = np.clip(mu_train + bu[tu] + bi[ti] + (U[tu] * I[ti]).sum(1), 1.0, 5.0)
    metrics["LightGCN"] = {
        "rmse": float(np.sqrt(mean_squared_error(test_raw["rating"], lg_pred))),
        "mae": float(mean_absolute_error(test_raw["rating"], lg_pred)),
    }

    def lgcn_score_user(uid, item_idx):
        uu = user_to_idx.get(uid)
        if uu is None:
            return np.full(len(item_idx), mu_train)
        return np.clip(mu_train + bu[uu] + bi[item_idx] + U[uu] @ I[item_idx].T, 1.0, 5.0)

    # ---- ranking metrics (single seed, 200 users) ----
    print("Computing ranking metrics...")
    for name, fn in [("MF", mf_score_user), ("CatBoost", cb_score_user),
                     ("LightGCN", lgcn_score_user)]:
        p, r, nd = ranking_metrics(fn, test_raw, train_raw, movie_id_arr)
        metrics[name].update({"p10": p, "r10": r, "ndcg10": nd})
        print(f"  {name:10} RMSE {metrics[name]['rmse']:.4f}  "
              f"P@10 {p:.4f}  R@10 {r:.4f}  NDCG@10 {nd:.4f}")

    # ---- shared metadata + per-user history ----
    print("Saving metadata and history...")
    movies_out = movies_feat[["movie_id", "title", "year", "genres"]].copy()
    movies_out["year"] = movies_out["year"].astype("Int64")
    movies_out.to_parquet(ART / "movies.parquet", index=False)

    users_out = users[["user_id", "gender", "age", "occupation"]].copy()
    users_out["age_band"] = users_out["age"].map(AGE_LABELS)
    users_out["occupation_name"] = users_out["occupation"].map(OCC_MAP)
    users_out.to_parquet(ART / "users.parquet", index=False)

    train_seen = train_raw.groupby("user_id")["movie_id"].apply(list).to_dict()
    test_rel = (test_raw[test_raw["rating"] >= 4.0]
                .groupby("user_id")["movie_id"].apply(list).to_dict())
    # per-user mean rating in train, for display
    user_train_mean = train_raw.groupby("user_id")["rating"].mean().to_dict()

    hist_users = np.array(all_user_ids)
    np.savez(
        ART / "history.npz",
        user_ids=hist_users,
        seen=np.array([json.dumps(train_seen.get(u, [])) for u in all_user_ids], dtype=object),
        relevant=np.array([json.dumps(test_rel.get(u, [])) for u in all_user_ids], dtype=object),
        train_mean=np.array([user_train_mean.get(u, mu_train) for u in all_user_ids]),
    )

    metrics["dataset"] = {
        "n_users": n_u, "n_movies": n_m, "n_ratings": int(len(ratings)),
        "train": int(len(train_raw)), "val": int(len(val_raw)), "test": int(len(test_raw)),
        "eval_users": 200, "eval_seed": 42,
    }
    with open(ART / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Done. Artifacts written to", ART)


if __name__ == "__main__":
    main()
