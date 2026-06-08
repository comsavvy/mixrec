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

An interactive demo of the recommender models from the AIMS Rwanda thesis
*"Mixed-Type Recommender Systems"*, trained on **MovieLens-1M**. It lets you
explore and compare three different modelling families side by side:

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

## Deployment

This repository is ready to deploy as-is on either host:

- **Hugging Face Spaces** - create a Space with the *Streamlit* SDK and push this
  repo. The YAML header at the top of this README configures it automatically.
- **Streamlit Community Cloud** - point [share.streamlit.io](https://share.streamlit.io)
  at this repo with `app.py` as the entrypoint. (The YAML header above is ignored.)

## Retraining (optional)

To regenerate `artifacts/` from scratch you need the MovieLens-1M `.dat` files
and the training-only extras:

```bash
pip install -r requirements.txt -r requirements-precompute.txt
# place ratings.dat / users.dat / movies.dat in ../datasets/
python precompute.py
```

## Project layout

```
app.py                       Streamlit UI (pages, dark mode, charts)
recommender.py               Loads artifacts, scores top-K live
theme.py                     Logo SVG, CSS, HTML renderers
precompute.py                Training script (regenerates artifacts/)
artifacts/                   Trained models + cached frames (committed)
images/                      Thesis figures used in the Interpretability page
.streamlit/config.toml       Theme + server config
requirements.txt             Runtime dependencies
requirements-precompute.txt  Extra deps for retraining only
```
