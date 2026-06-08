"""
MixRec - demo website for the AIMS Rwanda thesis "Mixed-Type Recommender
Systems". Self-contained: trained artifacts ship in artifacts/, so the app
runs with no precompute step.

    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st

import theme
from recommender import (
    Recommender, MODELS, MODEL_LABELS, MODEL_SHORT, MODEL_TAGLINE,
    EMBEDDING_MODELS, AGE_OPTIONS, OCC_OPTIONS, artifacts_exist,
)

HERE = Path(__file__).resolve().parent
# Prefer a local images/ folder (standalone deploy); fall back to the thesis
# tree (Research/images) when running from inside the research repository.
IMAGES = HERE / "images"
if not IMAGES.exists():
    IMAGES = HERE.parent.parent / "images"

st.set_page_config(
    page_title="MixRec · Mixed-Type Recommender Systems",
    page_icon=theme.favicon_path(HERE),
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained models...")
def get_recommender():
    return Recommender()


def fmt_metric(value):
    return "-" if value is None else f"{value:.4f}"


def render_cards(df, kind="rec", reasons=None):
    """Render a recommendation or similarity DataFrame as HTML cards.

    reasons: optional list of short 'why' strings aligned with df rows.
    """
    rows = []
    for i, r in enumerate(df.itertuples(), start=1):
        if kind == "rec":
            why = reasons[i - 1] if reasons and i - 1 < len(reasons) else ""
            rows.append(theme.rec_card_html(i, r.title, r.genres, float(r.score), why))
        else:
            rows.append(theme.sim_card_html(i, r.title, r.genres, float(r.similarity)))
    st.markdown("".join(rows), unsafe_allow_html=True)


_BRAND_RANGE = ["#7C5CFF", "#4DA3FF", "#34D399", "#F5A623"]


def _chart_theme(chart):
    """Apply light/dark colours to an Altair chart so it follows the toggle."""
    dark = st.session_state.get("dark_mode", False)
    fg = "#ECEAF5" if dark else "#1A1726"
    grid = "#322D47" if dark else "#E7E3F5"
    bg = "#1E1B29" if dark else "#FFFFFF"
    return (
        chart.properties(background=bg, height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=fg, titleColor=fg, gridColor=grid,
                        domainColor=grid, tickColor=grid)
        .configure_legend(labelColor=fg, titleColor=fg)
    )


def _bar_chart(df):
    """Themed stacked bar chart from a model-indexed metrics DataFrame."""
    idx = df.index.name or "index"
    long = df.reset_index(names=idx).melt(id_vars=idx, var_name="Metric",
                                          value_name="Value")
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X(f"{idx}:N", title=None, sort=list(df.index), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Value:Q", title=None),
        color=alt.Color("Metric:N", title=None,
                        scale=alt.Scale(range=_BRAND_RANGE)),
        tooltip=[idx, "Metric", alt.Tooltip("Value:Q", format=".4f")],
    )
    return _chart_theme(chart)


def _line_chart(df, x_title="K", y_title="NDCG@K"):
    """Themed multi-series line chart from a K-indexed DataFrame."""
    idx = df.index.name or "index"
    long = df.reset_index(names=idx).melt(id_vars=idx, var_name="Model",
                                          value_name="Value")
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X(f"{idx}:Q", title=x_title),
        y=alt.Y("Value:Q", title=y_title),
        color=alt.Color("Model:N", title=None, scale=alt.Scale(range=_BRAND_RANGE)),
        tooltip=[idx, "Model", alt.Tooltip("Value:Q", format=".4f")],
    )
    return _chart_theme(chart)


def _filter_controls(rec, key_prefix):
    """Shared genre + decade filter widgets. Returns (genres, min_year, max_year)."""
    lo, hi = rec.year_bounds()
    yr_key = f"{key_prefix}_years"
    st.session_state.setdefault(yr_key, (lo, hi))
    with st.expander("Filter results (optional)"):
        genres = st.multiselect("Limit to genres", rec.genres_list(),
                                key=f"{key_prefix}_genres")
        yr = st.slider("Release-year range", lo, hi, key=yr_key)
    min_year = yr[0] if yr[0] > lo else None
    max_year = yr[1] if yr[1] < hi else None
    return (genres or None), min_year, max_year


def page_recommend(rec):
    heading = st.empty()  # filled once the slider value is known
    st.caption("Pick a user from MovieLens-1M and compare what each model "
               "recommends. Movies the user already rated in the training "
               "period are excluded from every list.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        samples = rec.sample_user_ids(n=12, seed=7)
        # selecting an example user updates the number box via a callback, so
        # the two stay in sync without clashing with persisted session state
        if "rec_user_id" not in st.session_state:
            st.session_state["rec_user_id"] = int(samples[0])

        def _sync_user_id():
            st.session_state["rec_user_id"] = int(st.session_state["rec_user_select"])

        st.session_state.setdefault("rec_user_select", samples[0])
        st.selectbox("Example users", samples,
                     key="rec_user_select", on_change=_sync_user_id)
        user_id = st.number_input(
            "or enter any user id (1 - 6040)",
            min_value=int(rec.user_ids.min()),
            max_value=int(rec.user_ids.max()),
            step=1,
            key="rec_user_id",
        )
        st.session_state.setdefault("rec_k", 10)
        k = st.slider("How many recommendations", 5, 20, key="rec_k")
        st.session_state.setdefault("rec_explain", False)
        explain = st.toggle("Explain each pick", key="rec_explain",
                            help="Show why each model recommended a title.")

    heading.subheader(f"Top-{k} recommendations")

    if not rec.known_user(user_id):
        st.warning("That user id is not in the dataset.")
        return

    profile = rec.user_profile(user_id)
    with col_b:
        st.markdown(theme.profile_html(user_id, profile), unsafe_allow_html=True)

    liked = rec.liked_history(user_id, k=12)
    liked_ids = [m for m, _, _ in liked]
    with st.expander("A few films this user rated during training"):
        if liked:
            st.dataframe(
                pd.DataFrame(liked, columns=["movie_id", "title", "genres"]),
                hide_index=True, width="stretch",
            )
        else:
            st.write("No training history recorded for this user.")

    genres, min_year, max_year = _filter_controls(rec, "rec")

    st.divider()
    t0 = time.perf_counter()
    recs = {m: rec.recommend_filtered(user_id, m, k=k, genres=genres,
                                      min_year=min_year, max_year=max_year)
            for m in MODELS}
    elapsed_ms = (time.perf_counter() - t0) * 1000
    st.markdown(theme.badge_html(f"scored {len(MODELS)} models live in "
                                 f"{elapsed_ms:.0f} ms"), unsafe_allow_html=True)

    cols = st.columns(len(MODELS), gap="medium")
    for col, model in zip(cols, MODELS):
        with col:
            st.markdown(
                theme.model_head_html(
                    MODEL_SHORT[model], MODEL_TAGLINE[model],
                    metric=rec.headline_metric(model, k), k=k),
                unsafe_allow_html=True,
            )
            reasons = _reasons_for(rec, model, recs[model], user_id, liked_ids) \
                if explain else None
            render_cards(recs[model], kind="rec", reasons=reasons)

    # download the combined recommendations
    combined = pd.concat(
        [d.assign(model=MODEL_SHORT[m]) for m, d in recs.items()],
        ignore_index=True)[["model", "title", "genres", "score"]]
    st.download_button("Download these recommendations (CSV)",
                       combined.to_csv(index=False).encode(),
                       file_name=f"recommendations_user_{int(user_id)}.csv",
                       mime="text/csv")

    st.divider()
    st.subheader("How much do the models agree?")
    st.caption("Number of titles shared between each pair of lists for this "
               "user. The models often rank quite differently even when their "
               "overall metrics are close.")
    overlap, _ = rec.overlap_matrix(user_id, k=k)
    overlap.index = [MODEL_SHORT[m] for m in overlap.index]
    overlap.columns = [MODEL_SHORT[m] for m in overlap.columns]
    st.markdown(theme.stat_table_html(overlap, fmt="{:.0f}", gradient=True),
                unsafe_allow_html=True)


def _reasons_for(rec, model, df, user_id, liked_ids):
    """Short 'why' string per recommended row for the given model."""
    reasons = []
    for r in df.itertuples():
        mid = int(r.movie_id)
        if model == "CatBoost":
            contribs = rec.explain_catboost(int(user_id), mid, top=2)
            pos = [name for name, c in contribs if c > 0]
            reasons.append("driven by " + ", ".join(pos) if pos
                           else "balanced feature signal")
        else:
            near = rec.explain_embedding(liked_ids, mid, model=model, top=1)
            reasons.append(f"because you liked {near[0]}" if near
                           else "fits your overall taste")
    return reasons


def page_cold_start(rec):
    st.subheader("Build your taste")
    st.caption("No account needed. Tell us a handful of films you enjoy and a "
               "little about yourself, and the models will recommend for you "
               "live — this is the cold-start case that separates the three "
               "approaches.")

    options = rec.movie_options()
    label_to_id = dict(options)
    default = [lbl for lbl, _ in options[:3]]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.session_state.setdefault("cold_picks", default)
        picks = st.multiselect("Films you like (most rated first)",
                               [lbl for lbl, _ in options],
                               key="cold_picks")
    with c2:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True,
                          key="cold_gender")
        st.session_state.setdefault("cold_age", list(AGE_OPTIONS.keys())[2])
        age = st.selectbox("Age group", list(AGE_OPTIONS.keys()),
                           key="cold_age")
        st.session_state.setdefault("cold_occ", list(OCC_OPTIONS.keys())[12])
        occ = st.selectbox("Occupation", list(OCC_OPTIONS.keys()),
                           key="cold_occ")

    st.session_state.setdefault("cold_k", 10)
    k = st.slider("How many recommendations", 5, 20, key="cold_k")
    st.session_state.setdefault("cold_explain", False)
    explain = st.toggle("Explain each pick", key="cold_explain",
                        help="Show why each model recommended a title.")
    genres, min_year, max_year = _filter_controls(rec, "cold")

    liked_ids = [label_to_id[p] for p in picks]
    if len(liked_ids) == 0:
        st.info("Pick at least one film to get recommendations.")
        return

    st.divider()
    t0 = time.perf_counter()
    recs = rec.cold_start_recommend(
        liked_ids, gender, AGE_OPTIONS[age], OCC_OPTIONS[occ], k=k,
        genres=genres, min_year=min_year, max_year=max_year)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    st.markdown(
        theme.badge_html(f"built a fresh user and scored {len(MODELS)} models in "
                         f"{elapsed_ms:.0f} ms"), unsafe_allow_html=True)
    st.caption("Matrix Factorisation and LightGCN place you in their learned "
               "space by folding in your liked films; CatBoost builds a feature "
               "row from your taste and demographics. Notice MF/LightGCN lean to "
               "popular neighbours while CatBoost leans on film quality.")

    cols = st.columns(len(MODELS), gap="medium")
    for col, model in zip(cols, MODELS):
        with col:
            st.markdown(
                theme.model_head_html(MODEL_SHORT[model], MODEL_TAGLINE[model]),
                unsafe_allow_html=True)
            reasons = None
            if explain and model == "CatBoost":
                urow = _cold_row(rec, gender, age, occ, liked_ids)
                reasons = []
                for r in recs[model].itertuples():
                    contribs = rec.explain_catboost(urow, int(r.movie_id), top=2)
                    pos = [n for n, c in contribs if c > 0]
                    reasons.append("driven by " + ", ".join(pos) if pos
                                   else "balanced signal")
            elif explain:
                reasons = [
                    (f"because you liked {n[0]}"
                     if (n := rec.explain_embedding(liked_ids, int(r.movie_id),
                                                    model=model, top=1)) else
                     "fits your taste")
                    for r in recs[model].itertuples()]
            render_cards(recs[model], kind="rec", reasons=reasons)


def _cold_row(rec, gender, age_label, occ_label, liked_ids):
    import numpy as np
    liked_idx = np.array([rec.movie_to_idx[int(m)] for m in liked_ids
                          if int(m) in rec.movie_to_idx], dtype=int)
    return rec._cold_user_row_cb(gender, AGE_OPTIONS[age_label],
                                 OCC_OPTIONS[occ_label], liked_idx)


def page_similar(rec):
    st.subheader("Find similar movies")
    st.caption("Pick a film and see its nearest neighbours in each model's "
               "learned item-embedding space (cosine similarity). CatBoost is "
               "a tree ensemble with no item embedding, so only the two "
               "embedding models appear here.")

    options = rec.movie_options()
    labels = [lbl for lbl, _ in options]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.setdefault("sim_movie", labels[0])
        pick = st.selectbox("Movie (most rated first)", labels,
                            key="sim_movie")
    with col2:
        st.session_state.setdefault("sim_k", 8)
        k = st.slider("Neighbours", 5, 15, key="sim_k")
    movie_id = dict(options)[pick]

    genres = rec.movie_genres(movie_id).replace("|", " · ")
    st.markdown(f"**Selected:** {rec.movie_title(movie_id)} &nbsp; "
                f"<span class='note'>{genres}</span>", unsafe_allow_html=True)

    st.divider()
    cols = st.columns(len(EMBEDDING_MODELS), gap="medium")
    for col, model in zip(cols, EMBEDDING_MODELS):
        with col:
            st.markdown(
                theme.model_head_html(MODEL_SHORT[model], MODEL_TAGLINE[model]),
                unsafe_allow_html=True,
            )
            render_cards(rec.similar_movies(movie_id, model=model, k=k), kind="sim")


def page_results(rec):
    st.subheader("Model comparison")
    st.caption("All models are evaluated under identical conditions: the same "
               "time-based 80/10/10 split, the same test set, and the same "
               "ranking-metric implementation.")

    m = rec.metrics
    ds = m.get("dataset", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{ds.get('n_users', 0):,}")
    c2.metric("Movies", f"{ds.get('n_movies', 0):,}")
    c3.metric("Ratings", f"{ds.get('n_ratings', 0):,}")
    c4.metric("Eval users / seed", ds.get("eval_users", "-"))

    rows = []
    for model in MODELS:
        d = m.get(model, {})
        rows.append({
            "Model": MODEL_LABELS[model],
            "RMSE": d.get("rmse"), "MAE": d.get("mae"),
            "Precision@10": d.get("p10"), "Recall@10": d.get("r10"),
            "NDCG@10": d.get("ndcg10"),
        })
    table = pd.DataFrame(rows).set_index("Model")
    st.markdown(
        theme.stat_table_html(
            table, fmt="{:.4f}",
            better={"RMSE": "min", "MAE": "min", "Precision@10": "max",
                    "Recall@10": "max", "NDCG@10": "max"}),
        unsafe_allow_html=True)
    st.caption("Green marks the best model on each metric (lowest error, "
               "highest ranking score).")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Ranking quality", "Rating accuracy", "NDCG vs K",
         "Diversity & bias", "Thesis figure"])
    short = [MODEL_SHORT[mm] for mm in MODELS]
    with tab1:
        rank_df = pd.DataFrame(
            {"Precision@10": [m[mm].get("p10") for mm in MODELS],
             "Recall@10": [m[mm].get("r10") for mm in MODELS],
             "NDCG@10": [m[mm].get("ndcg10") for mm in MODELS]},
            index=short)
        st.altair_chart(_bar_chart(rank_df), theme=None, width="stretch")
    with tab2:
        acc_df = pd.DataFrame(
            {"RMSE": [m[mm].get("rmse") for mm in MODELS],
             "MAE": [m[mm].get("mae") for mm in MODELS]},
            index=short)
        st.altair_chart(_bar_chart(acc_df), theme=None, width="stretch")
    with tab3:
        st.markdown("How ranking quality changes as the list grows. Computed "
                    "live on the held-out test set, so every point uses the "
                    "same protocol as the headline numbers.")
        curve = rec.metric_curve(ks=range(1, 21))
        st.altair_chart(_line_chart(curve), theme=None, width="stretch")
        st.caption("X axis: list length K (1-20). Y axis: mean NDCG@K.")
    with tab4:
        st.markdown("Beyond accuracy: how much of the catalogue each model "
                    "reaches, how popular its picks are, and how varied each "
                    "list is. This exposes popularity bias.")
        stats = rec.catalogue_stats(k=10)
        st.markdown(theme.stat_table_html(stats, fmt="{:.3f}", gradient=True),
                    unsafe_allow_html=True)
        st.caption("Catalogue coverage: share of all films ever recommended. "
                   "Avg popularity: 0 = obscure, 1 = blockbuster. "
                   "List diversity: higher means more genre variety per list.")
    with tab5:
        img = IMAGES / "model_comparison.png"
        if img.exists():
            st.image(str(img), caption="Comparison figure from the thesis "
                     "notebook.", width="stretch")


def page_interpretability(rec):
    st.subheader("Interpretability")
    st.caption("Summary metrics tell us which model wins on each axis, but not "
               "why. These two diagnostics show what the models actually "
               "learned from the mixed-type input.")

    tab1, tab2 = st.tabs(["CatBoost feature importance", "LightGCN embeddings"])
    with tab1:
        st.markdown("How much each feature contributes to loss reduction across "
                    "the boosted ensemble. This shows whether the engineered "
                    "statistics pull weight against the raw side information.")
        fi = IMAGES / "catboost_feature_importance.png"
        if fi.exists():
            st.image(str(fi), width="stretch")
    with tab2:
        st.markdown("The 64-dimensional item embeddings projected to 2D via PCA "
                    "and coloured by primary genre. Clustering by genre is a "
                    "visual test of whether the graph captured semantic "
                    "structure through message passing.")
        emb = IMAGES / "lightgcn_item_embeddings.png"
        if emb.exists():
            st.image(str(emb), width="stretch")


def page_about(rec):
    st.subheader("About this work")
    st.markdown(
        """
**Mixed-Type Recommender Systems: Implementation on MovieLens-1M**

- **Author:** Olusola Timothy Ogundepo
- **Supervisor:** Professor Ernest Fokoué
- **Institution:** African Institute for Mathematical Sciences (AIMS), Rwanda

The central question is how different modelling families handle datasets that
mix numerical, categorical and binary features alongside user-item interaction
data. Three models are compared:

| Model | Approach | Mixed-type handling |
|---|---|---|
| Matrix Factorisation | Collaborative filtering baseline | None, ratings only |
| CatBoost | Tree-based gradient boosting | Native categorical splits |
| LightGCN | Graph neural network | Feature-augmented init plus graph propagation |

**Split strategy:** time-based. Train on the oldest 80% of ratings, validate on
the next 10%, test on the most recent 10%, so a model always predicts future
preferences from past behaviour.

**Metrics:** RMSE and MAE for rating accuracy; Precision@10, Recall@10 and
NDCG@10 for top-K ranking quality.

This site loads models that were trained offline and scores your
recommendations live, so results appear instantly.
        """
    )


PAGES = {
    "Try the recommender": page_recommend,
    "Build your taste": page_cold_start,
    "Find similar movies": page_similar,
    "Model comparison": page_results,
    "Interpretability": page_interpretability,
    "About": page_about,
}


def main():
    st.markdown(theme.CSS, unsafe_allow_html=True)

    # Keep widget selections alive when the user leaves a page and comes back.
    # Streamlit discards the state of widgets that are not rendered in a run;
    # re-assigning each stored key marks it active so nothing resets to default.
    for _k in list(st.session_state.keys()):
        st.session_state[_k] = st.session_state[_k]

    # Dark mode: read the persisted toggle and overlay the dark stylesheet on
    # top of the base CSS before anything renders, so the whole page is dark.
    st.session_state.setdefault("dark_mode", False)
    if st.session_state["dark_mode"]:
        st.markdown(theme.DARK_CSS, unsafe_allow_html=True)

    if not artifacts_exist():
        st.markdown(theme.header_html(), unsafe_allow_html=True)
        st.error(
            "Model artifacts were not found. Run the one-time training step "
            "first:\n\n```\nconda activate RSProj\ncd webapp\npython precompute.py\n```"
        )
        st.stop()

    rec = get_recommender()

    pages = list(PAGES.keys())
    # Restore the active page from the URL (?page=...) so a reload stays put,
    # both locally and when hosted. Streamlit resets widget state on a full
    # reload, but query params persist in the URL.
    qp_page = st.query_params.get("page")
    if "nav" not in st.session_state and qp_page in pages:
        st.session_state["nav"] = qp_page

    with st.sidebar:
        st.markdown(theme.sidebar_brand_html(), unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate", pages, key="nav")
        st.divider()
        st.toggle("Dark mode", key="dark_mode")
        st.caption("MovieLens-1M · time-based 80/10/10 split. "
                   "Models trained offline, served live.")

    if st.query_params.get("page") != page:
        st.query_params["page"] = page

    st.markdown(theme.header_html(), unsafe_allow_html=True)
    PAGES[page](rec)


if __name__ == "__main__":
    main()
