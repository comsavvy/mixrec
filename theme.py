"""
Visual assets for the demo: a custom SVG logo, global CSS, and HTML renderers
for recommendation and similar-movie cards. Keeping these here keeps app.py
focused on layout and logic.
"""

import html

BRAND = "MixRec"
TAGLINE = "Mixed-Type Recommender Systems"

# Custom logo: a small graph (three connected nodes) with a recommendation
# spark, drawn in the app's violet gradient. Replaces the institutional logo.
LOGO_ICON = """
<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MixRec logo">
  <defs>
    <linearGradient id="mixg" x1="0" y1="0" x2="48" y2="48"
                    gradientUnits="userSpaceOnUse">
      <stop stop-color="#7C5CFF"/>
      <stop offset="1" stop-color="#4DA3FF"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="44" height="44" rx="12" fill="url(#mixg)"/>
  <path d="M16 16 L32 24 M16 32 L32 24" stroke="#FFFFFF" stroke-width="2.4"
        stroke-linecap="round" opacity="0.9"/>
  <circle cx="16" cy="16" r="4.4" fill="#FFFFFF"/>
  <circle cx="16" cy="32" r="4.4" fill="#FFFFFF"/>
  <circle cx="33" cy="24" r="5.4" fill="#FFFFFF"/>
  <path d="M33 20.4 l1.05 2.13 2.35.34-1.7 1.66.4 2.34L33 25.9l-2.1 1.1.4-2.34
           -1.7-1.66 2.35-.34Z" fill="#6C4DF6"/>
</svg>
"""


def logo_html(size=40):
    return LOGO_ICON.format(size=size)


def favicon_path(directory):
    """Write the MixRec logo to an SVG file in `directory` and return its path.

    Lets the browser tab favicon reuse the same brand mark as the sidebar logo
    instead of a generic emoji. Streamlit's page_icon accepts a local file path.
    """
    from pathlib import Path

    path = Path(directory) / "favicon.svg"
    svg = LOGO_ICON.format(size=48).strip()
    if not path.exists() or path.read_text(encoding="utf-8") != svg:
        path.write_text(svg, encoding="utf-8")
    return str(path)


def sidebar_brand_html():
    return f"""
<div class="brand">
  {logo_html(38)}
  <div class="brand-text">
    <div class="brand-name">{BRAND}</div>
    <div class="brand-sub">{html.escape(TAGLINE)}</div>
  </div>
</div>
"""


def header_html():
    return f"""
<div class="app-header">
  <div class="app-header-icon">{logo_html(54)}</div>
  <div>
    <div class="app-title">{BRAND}</div>
    <div class="app-subtitle">{html.escape(TAGLINE)} · MovieLens-1M</div>
  </div>
</div>
"""


CSS = """
<style>
:root {
  --mix-grad: linear-gradient(135deg, #7C5CFF 0%, #4DA3FF 100%);
  --mix-violet: #6C4DF6;
  --mix-ink: #1A1726;
  --mix-muted: #6B6680;
  --mix-line: #E7E3F5;
  --mix-surface: #FFFFFF;     /* card / item background */
  --mix-surface-2: #FBFAFF;   /* profile panel background */
  --mix-chip-bg: #F1ECFF;     /* genre chip / metric pill background */
  --mix-badge-bg: #E6F6EC;    /* latency badge background */
  --mix-badge-ink: #1f9d55;   /* latency badge text */
  --mix-star-off: #DAD6EA;    /* empty star colour */
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }

/* sidebar brand lockup */
.brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .4rem; }
.brand-text .brand-name { font-weight: 800; font-size: 1.25rem; letter-spacing: -.01em;
  background: var(--mix-grad); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; line-height: 1; }
.brand-text .brand-sub { font-size: .72rem; color: var(--mix-muted); margin-top: 2px; }

/* main header banner */
.app-header { display: flex; align-items: center; gap: 1rem; padding: 1.1rem 1.3rem;
  border-radius: 18px; background: var(--mix-grad); color: #fff; margin-bottom: 1.1rem;
  box-shadow: 0 10px 30px rgba(108,77,246,.25); }
.app-header-icon { background: rgba(255,255,255,.16); border-radius: 14px; padding: 6px;
  display: flex; }
.app-title { font-size: 1.65rem; font-weight: 800; line-height: 1.05; letter-spacing: -.02em; }
.app-subtitle { font-size: .9rem; opacity: .92; margin-top: 2px; }

/* model column header */
.model-head { border-bottom: 2px solid var(--mix-line); padding-bottom: .5rem;
  margin-bottom: .7rem; }
.model-head .mh-name { font-weight: 700; font-size: 1.05rem; color: var(--mix-ink); }
.model-head .mh-tag { font-size: .76rem; color: var(--mix-muted); }
.model-head .mh-metric { display: inline-block; margin-top: .35rem; font-size: .72rem;
  font-weight: 600; color: var(--mix-violet); background: var(--mix-chip-bg); border-radius: 999px;
  padding: 2px 9px; }

/* recommendation / similarity card */
.rec-card { display: flex; align-items: center; gap: .7rem; padding: .55rem .65rem;
  border: 1px solid var(--mix-line); border-radius: 12px; margin-bottom: .5rem;
  background: var(--mix-surface); transition: box-shadow .15s ease, transform .15s ease; }
.rec-card:hover { box-shadow: 0 6px 18px rgba(26,23,38,.08); transform: translateY(-1px); }
.rec-rank { flex: 0 0 26px; height: 26px; width: 26px; border-radius: 8px;
  background: var(--mix-grad); color: #fff; font-weight: 700; font-size: .82rem;
  display: flex; align-items: center; justify-content: center; }
.rec-body { flex: 1 1 auto; min-width: 0; }
.rec-title { font-weight: 600; font-size: .92rem; color: var(--mix-ink);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rec-genres { margin-top: 3px; display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font-size: .66rem; color: var(--mix-violet); background: var(--mix-chip-bg);
  border-radius: 6px; padding: 1px 7px; white-space: nowrap; }
.why { margin-top: 4px; font-size: .72rem; color: var(--mix-muted); font-style: italic;
  line-height: 1.25; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden; overflow-wrap: anywhere; }
.lat-badge { display: inline-block; font-size: .72rem; font-weight: 600; color: var(--mix-badge-ink);
  background: var(--mix-badge-bg); border-radius: 999px; padding: 2px 10px; }
.rec-score { flex: 0 0 auto; text-align: right; min-width: 56px; }
.rec-score-val { font-weight: 700; font-size: .9rem; color: var(--mix-ink); }
.stars { font-size: .72rem; letter-spacing: 1px; line-height: 1; }
.stars .on { color: #F5A623; }
.stars .off { color: var(--mix-star-off); }
.sim-val { font-weight: 700; font-size: .82rem; color: var(--mix-violet); }

/* legend / small note */
.note { color: var(--mix-muted); font-size: .82rem; }

/* user profile card */
.profile { border: 1px solid var(--mix-line); border-radius: 14px; padding: .85rem 1rem;
  background: var(--mix-surface-2); }
.pf-head { font-weight: 700; color: var(--mix-ink); margin-bottom: .55rem; }
.pf-grid { display: flex; flex-wrap: wrap; gap: .55rem; }
.pf-item { background: var(--mix-surface); border: 1px solid var(--mix-line); border-radius: 10px;
  padding: .35rem .6rem; min-width: 92px; }
.pf-label { font-size: .68rem; color: var(--mix-muted); text-transform: uppercase;
  letter-spacing: .03em; }
.pf-value { font-size: .9rem; font-weight: 600; color: var(--mix-ink); margin-top: 1px; }

/* themed data table (used instead of st.dataframe so it follows dark mode) */
.mix-table { width: 100%; border-collapse: collapse; font-size: .86rem;
  margin: .1rem 0 .3rem; }
.mix-table th, .mix-table td { padding: .5rem .7rem; text-align: right;
  border-bottom: 1px solid var(--mix-line); white-space: nowrap; }
.mix-table thead th { color: var(--mix-muted); font-weight: 600; font-size: .77rem; }
.mix-table td:first-child, .mix-table th:first-child { text-align: left;
  color: var(--mix-ink); font-weight: 600; }
.mix-table tbody td { color: var(--mix-ink); }
.mix-table tbody tr:hover td { background: var(--mix-chip-bg); }
.mix-table .best { color: var(--mix-badge-ink); font-weight: 700; }
</style>
"""


# Dark-mode override. Injected after CSS when the sidebar toggle is on. It flips
# the brand variables and restyles Streamlit's own chrome (app background,
# sidebar, text, inputs, expanders, tabs, tables) so the whole page goes dark.
DARK_CSS = """
<style>
:root, .stApp {
  --mix-violet: #A48CFF;
  --mix-ink: #ECEAF5;
  --mix-muted: #A39DBB;
  --mix-line: #322D47;
  --mix-surface: #1E1B29;
  --mix-surface-2: #232031;
  --mix-chip-bg: #2C2546;
  --mix-badge-bg: #14331E;
  --mix-badge-ink: #5FD08B;
  --mix-star-off: #45405C;
}

/* app shell */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"] { background-color: #131120 !important; }
[data-testid="stSidebar"] { background-color: #1A1726 !important;
  border-right: 1px solid #2A2640; }
[data-testid="stHeader"] { color: #ECEAF5; }

/* base text */
.stApp, .stMarkdown, [data-testid="stMarkdownContainer"],
p, li, label, [data-testid="stWidgetLabel"] *,
h1, h2, h3, h4, h5, h6 { color: #ECEAF5 !important; }
[data-testid="stCaptionContainer"], .note, small { color: #A39DBB !important; }

/* inputs: select, multiselect, number, slider, radio surfaces */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
  background-color: #211D30 !important; border-color: #322D47 !important;
  color: #ECEAF5 !important; }
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg { fill: #A39DBB; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background-color: #211D30 !important; color: #ECEAF5 !important; }
[data-baseweb="tag"] { background-color: #2C2546 !important; color: #D9CCFF !important; }

/* expanders, tabs, dataframes, tables */
[data-testid="stExpander"] details { background-color: #1E1B29 !important;
  border: 1px solid #322D47 !important; border-radius: 12px; }
[data-testid="stExpander"] summary { color: #ECEAF5 !important; }
[data-baseweb="tab-list"] { background-color: transparent; }
[data-baseweb="tab"] { color: #A39DBB; }
[data-baseweb="tab"][aria-selected="true"] { color: #ECEAF5; }
[data-testid="stTable"] td, [data-testid="stTable"] th {
  color: #ECEAF5 !important; border-color: #322D47 !important; }
[data-testid="stDataFrame"] { background-color: #1E1B29; border-radius: 10px; }

/* download / generic buttons */
[data-testid="stDownloadButton"] button, [data-testid="stBaseButton-secondary"] {
  background-color: #211D30 !important; color: #ECEAF5 !important;
  border-color: #322D47 !important; }

/* dividers */
hr { border-color: #2A2640 !important; }
</style>
"""


def _stars(score, out_of=5):
    """Five glyphs filled to the rounded half-star nearest to score/out_of*5."""
    filled = int(round((score / out_of) * 5))
    filled = max(0, min(5, filled))
    on = '<span class="on">' + "★" * filled + "</span>"
    off = '<span class="off">' + "★" * (5 - filled) + "</span>"
    return f'<span class="stars">{on}{off}</span>'


def _chips(genres, limit=3):
    parts = [g for g in str(genres).split("|") if g][:limit]
    return "".join(f'<span class="chip">{html.escape(g)}</span>' for g in parts)


def rec_card_html(rank, title, genres, score, reason=""):
    reason_html = (f'<div class="why">{html.escape(reason)}</div>' if reason else "")
    return f"""
<div class="rec-card">
  <div class="rec-rank">{rank}</div>
  <div class="rec-body">
    <div class="rec-title" title="{html.escape(str(title))}">{html.escape(str(title))}</div>
    <div class="rec-genres">{_chips(genres)}</div>
    {reason_html}
  </div>
  <div class="rec-score">
    <div class="rec-score-val">{score:.2f}</div>
    {_stars(score)}
  </div>
</div>
"""


def badge_html(text):
    return f'<span class="lat-badge">{html.escape(str(text))}</span>'


def sim_card_html(rank, title, genres, similarity):
    pct = max(0, min(100, int(round(similarity * 100))))
    return f"""
<div class="rec-card">
  <div class="rec-rank">{rank}</div>
  <div class="rec-body">
    <div class="rec-title" title="{html.escape(str(title))}">{html.escape(str(title))}</div>
    <div class="rec-genres">{_chips(genres)}</div>
  </div>
  <div class="rec-score">
    <div class="sim-val">{pct}%</div>
    <div class="note">match</div>
  </div>
</div>
"""


def model_head_html(name, tagline, metric=None, k=10):
    metric_html = (f'<div class="mh-metric">NDCG@{k} {metric:.3f}</div>'
                   if metric is not None else "")
    return f"""
<div class="model-head">
  <div class="mh-name">{html.escape(name)}</div>
  <div class="mh-tag">{html.escape(tagline)}</div>
  {metric_html}
</div>
"""


def profile_html(user_id, profile):
    """User profile as wrapping chips so long values (occupation) never clip."""
    fields = [
        ("Gender", profile["gender"]),
        ("Age band", profile["age_band"]),
        ("Occupation", profile["occupation"]),
        ("Movies rated", str(profile["n_rated_train"])),
        ("Avg rating", f"{profile['train_mean']:.2f}"
            if profile["train_mean"] == profile["train_mean"] else "-"),
    ]
    items = "".join(
        f'<div class="pf-item"><div class="pf-label">{html.escape(lbl)}</div>'
        f'<div class="pf-value">{html.escape(val)}</div></div>'
        for lbl, val in fields
    )
    return f"""
<div class="profile">
  <div class="pf-head">User {int(user_id)}</div>
  <div class="pf-grid">{items}</div>
</div>
"""


def stat_table_html(df, fmt="{:.4f}", better=None, gradient=False):
    """Render a DataFrame as a themed HTML table that follows light/dark mode.

    better:   optional {column: "max"|"min"} to bold-green the best cell.
    gradient: if True, shade each cell by its column-normalised value (violet),
              replacing st.dataframe's background_gradient.
    """
    better = better or {}
    cols = list(df.columns)
    best_idx = {}
    for c, direction in better.items():
        if c in df.columns:
            vals = df[c].astype(float)
            best_idx[c] = vals.idxmin() if direction == "min" else vals.idxmax()
    col_range = {}
    if gradient:
        for c in cols:
            v = df[c].astype(float)
            col_range[c] = (float(v.min()), float(v.max()))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows_html = []
    for name, row in df.iterrows():
        cells = [f"<td>{html.escape(str(name))}</td>"]
        for c in cols:
            val = row[c]
            ok = val == val  # not NaN
            txt = fmt.format(val) if ok else "-"
            style = ""
            if gradient and ok:
                lo, hi = col_range[c]
                a = 0.0 if hi == lo else (float(val) - lo) / (hi - lo)
                style = f' style="background: rgba(124,92,255,{0.10 + 0.42 * a:.2f})"'
            cls = ' class="best"' if c in best_idx and name == best_idx[c] else ""
            cells.append(f"<td{cls}{style}>{html.escape(txt)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    index_label = html.escape(str(df.index.name or ""))
    return (f'<table class="mix-table"><thead><tr><th>{index_label}</th>{head}</tr>'
            f'</thead><tbody>{"".join(rows_html)}</tbody></table>')

