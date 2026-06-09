---
title: MixRec
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# MixRec - Mixed-Type Recommender Systems

MixRec is an interactive movie-recommendation demo trained on **MovieLens-1M**.
It puts three different modelling families side by side so you can see how each
one recommends, compares, and explains its picks:

**▶ Live demo: [comsavvy-mixrec.streamlit.app](https://comsavvy-mixrec.streamlit.app/)**

| Model | Family | Strength |
|-------|--------|----------|
| **Matrix Factorisation** | Latent-factor collaborative filtering | Strong, balanced ranking |
| **CatBoost (extended, 42 feat)** | Gradient-boosted trees on mixed features | Best rating accuracy, low popularity bias |
| **LightGCN (feature-augmented)** | Graph neural network | Best ranking (NDCG@10) |

## Features

- **Try the recommender** - top-K picks for any user, scored live across all three models.
- **Build your taste** - cold-start: pick a few films you like and get recommendations.
- **Find similar movies** - nearest neighbours in the learned embedding space.
- **Model comparison** - accuracy, ranking, NDCG-vs-K curves, and diversity / popularity-bias metrics.
- **Interpretability** - CatBoost feature importance and LightGCN embedding visualisations.
- Per-recommendation **explanations**, genre/year **filters**, CSV export, and a **dark mode** toggle.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py        # opens at http://localhost:8501
```

The app is self-contained: the trained models live in `artifacts/` (~11 MB),
so there is **no training or download step** to serve it. Note that the
LightGCN and MF embeddings are precomputed into `artifacts/*.npz`, so PyTorch is
**not** required to run the demo.

## Retraining (optional)

The committed `artifacts/` are all you need to run MixRec. To rebuild them from
scratch, install the training extras, add the MovieLens-1M `.dat` files
(`ratings.dat`, `users.dat`, `movies.dat`) where `precompute.py` expects them,
and run it:

```bash
pip install -r requirements.txt -r requirements-precompute.txt
python precompute.py
```

## Project layout

```
app.py                       Streamlit UI (pages, dark mode, charts)
recommender.py               Loads artifacts, scores top-K live
theme.py                     Logo SVG, CSS, HTML renderers
precompute.py                Training script (regenerates artifacts/)
artifacts/                   Trained models + cached frames (committed)
images/                      Figures used in the Interpretability page
.streamlit/config.toml       Theme + server config
requirements.txt             Runtime dependencies
requirements-precompute.txt  Extra deps for retraining only
```
