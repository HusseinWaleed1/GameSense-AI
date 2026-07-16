import os
import re
import pickle
import joblib
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Optional heavy imports guarded so the file can at least be inspected
# even if the libraries / model files are not present yet.
try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ==========================================================
# 0. CONFIG — adjust these to match your actual dataset columns
# ==========================================================
DATA_DIR = "."  # change if your .pkl / .faiss files live elsewhere

PATHS = {
    "df": os.path.join(DATA_DIR, "final_df.pkl"),
    "tfidf_vectorizer": os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"),
    "tfidf_matrix": os.path.join(DATA_DIR, "tfidf_matrix.pkl"),
    "faiss_index": os.path.join(DATA_DIR, "game_index.faiss"),
    "st_model": os.path.join(DATA_DIR, "sentence_transformer_model"),
}

# Candidate column names — the app will auto-detect whichever exists
COLUMN_CANDIDATES = {
    "title": ["title", "name", "game_name", "Title", "Name"],
    "description": ["description", "short_description", "about", "summary"],
    "genres": ["genres", "genre", "Genres"],
    "tags": ["tags", "popular_tags", "Tags"],
}

HISTORY_LIMIT = 10
TOP_K = 5

# Minimum RAW (pre-normalization) similarity required for a result set to be considered
# meaningful. Min-max normalization always stretches the best available result to 100%,
# even if that result is a weak/nonsense match — these floors catch that case.
MIN_RAW_SEMANTIC_SCORE = 0.30   # cosine similarity floor (embeddings are normalized, range ~[-1,1])
MIN_RAW_TFIDF_SCORE = 0.05      # cosine similarity floor on the TF-IDF side


# ==========================================================
# 1. Helpers
# ==========================================================
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_text(text: str) -> str:
    """Light cleaning for a user query (mirrors preprocessing done on the corpus)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def as_list(value):
    """Normalize a genres/tags cell (could be a list, a comma-string, or NaN) into a list of strings."""
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if isinstance(value, str):
        # handles "Action, RPG, Indie" or "['Action', 'RPG']"
        cleaned = value.strip("[]")
        parts = re.split(r"[,\|;]", cleaned)
        return [p.strip(" '\"") for p in parts if p.strip(" '\"")]
    return []


# ==========================================================
# 2. Load resources (cached so it only happens once per session)
# ==========================================================
@st.cache_resource(show_spinner="Loading models and data...")
def load_resources():
    df = joblib.load(PATHS["df"])
    tfidf_vectorizer = joblib.load(PATHS["tfidf_vectorizer"])
    tfidf_matrix = joblib.load(PATHS["tfidf_matrix"])

    faiss_index = faiss.read_index(PATHS["faiss_index"]) if faiss else None
    model = SentenceTransformer(PATHS["st_model"]) if SentenceTransformer else None

    cols = {key: find_column(df, cands) for key, cands in COLUMN_CANDIDATES.items()}
    return df, tfidf_vectorizer, tfidf_matrix, faiss_index, model, cols


# ==========================================================
# 3. Search functions
# ==========================================================
def semantic_search(query, model, faiss_index, df, k=20):
    if model is None or faiss_index is None:
        return pd.DataFrame(), np.array([])
    query_vec = model.encode([clean_text(query)], normalize_embeddings=True)
    scores, idx = faiss_index.search(np.array(query_vec, dtype="float32"), k)
    idx = idx[0]
    scores = scores[0]
    valid = idx >= 0
    result_df = df.iloc[idx[valid]].copy()
    result_df["semantic_score"] = scores[valid]
    return result_df, scores[valid]


def tfidf_search(query, vectorizer, matrix, df, k=20):
    query_vec = vectorizer.transform([clean_text(query)])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    result_df = df.iloc[top_idx].copy()
    result_df["tfidf_score"] = sims[top_idx]
    return result_df


def hybrid_search(query, resources, k=20, alpha=0.6):
    """
    alpha weights semantic vs tfidf: final = alpha*semantic + (1-alpha)*tfidf
    Both score sets are min-max normalized before combining so they're comparable.
    """
    df, tfidf_vectorizer, tfidf_matrix, faiss_index, model, cols = resources

    sem_df, _ = semantic_search(query, model, faiss_index, df, k=k)
    tf_df = tfidf_search(query, tfidf_vectorizer, tfidf_matrix, df, k=k)

    merged = pd.concat([sem_df, tf_df], axis=0)
    merged = merged[~merged.index.duplicated(keep="first")]

    for col, default in [("semantic_score", 0.0), ("tfidf_score", 0.0)]:
        if col not in merged.columns:
            merged[col] = default
        merged[col] = merged[col].fillna(default)

    def norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    merged["semantic_norm"] = norm(merged["semantic_score"])
    merged["tfidf_norm"] = norm(merged["tfidf_score"])
    merged["match_score"] = alpha * merged["semantic_norm"] + (1 - alpha) * merged["tfidf_norm"]

    merged = merged.sort_values("match_score", ascending=False)

    # Quality gate: were the RAW (pre-normalization) scores actually good, or did
    # normalization just stretch a weak result up to 100%?
    best_raw_semantic = merged["semantic_score"].max() if len(merged) else 0
    best_raw_tfidf = merged["tfidf_score"].max() if len(merged) else 0
    has_meaningful_match = (best_raw_semantic >= MIN_RAW_SEMANTIC_SCORE) or (best_raw_tfidf >= MIN_RAW_TFIDF_SCORE)

    return merged.head(k), has_meaningful_match


# ==========================================================
# 4. Personalized re-ranking
# ==========================================================
def personalize_rerank(results, history, cols, boost=0.15):
    """Boost games that share genres/tags with the user's recent search history."""
    if not history:
        results["personalized_score"] = results["match_score"]
        return results.sort_values("personalized_score", ascending=False)

    history_terms = set()
    for q in history:
        history_terms.update(clean_text(q).split())

    genre_col, tag_col = cols.get("genres"), cols.get("tags")

    def overlap_score(row):
        shared = 0
        total_terms = 0
        for col in [genre_col, tag_col]:
            if col:
                items = [g.lower() for g in as_list(row.get(col, ""))]
                total_terms += len(items)
                shared += sum(1 for it in items if any(term in it for term in history_terms))
        return shared / total_terms if total_terms else 0

    results = results.copy()
    results["history_overlap"] = results.apply(overlap_score, axis=1)
    results["personalized_score"] = results["match_score"] + boost * results["history_overlap"]
    return results.sort_values("personalized_score", ascending=False)


# ==========================================================
# 5. Explainability
# ==========================================================
def explain_row(row, query, cols, lang="ar"):
    genre_col, tag_col = cols.get("genres"), cols.get("tags")
    query_terms = set(clean_text(query).split())

    shared_genres = [g for g in as_list(row.get(genre_col, "")) if g.lower() in query_terms] if genre_col else []
    shared_tags = [t for t in as_list(row.get(tag_col, "")) if t.lower() in query_terms] if tag_col else []

    reasons = []
    if lang == "en":
        if row.get("semantic_norm", 0) > 0.6:
            reasons.append("Description content closely matches what you're looking for")
        if shared_genres:
            reasons.append(f"Shared Genre: {', '.join(shared_genres[:3])}")
        if shared_tags:
            reasons.append(f"Shared Tags: {', '.join(shared_tags[:3])}")
        if row.get("history_overlap", 0) > 0:
            reasons.append("Similar to games you've searched for before")
        if not reasons:
            reasons.append("General text match with your search")
    else:
        if row.get("semantic_norm", 0) > 0.6:
            reasons.append("Description content closely matches what you're looking for")
        if shared_genres:
            reasons.append(f"Shared Genre: {', '.join(shared_genres[:3])}")
        if shared_tags:
            reasons.append(f"Shared Tags: {', '.join(shared_tags[:3])}")
        if row.get("history_overlap", 0) > 0:
            reasons.append("Similar to games you've searched for before")
        if not reasons:
            reasons.append("General text match with your search")

    return {
        "match_score": round(float(row.get("personalized_score", row.get("match_score", 0))) * 100, 1),
        "semantic_score": round(float(row.get("semantic_norm", 0)) * 100, 1),
        "shared_genres": shared_genres,
        "shared_tags": shared_tags,
        "reason": " • ".join(reasons),
    }


import io
import datetime as dt

# ==========================================================
# 6. Streamlit UI
# ==========================================================
st.set_page_config(page_title="GameSense AI", page_icon="🎮", layout="wide", initial_sidebar_state="expanded")

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []                    # list of past queries
if "history_log" not in st.session_state:
    st.session_state.history_log = []                 # [(timestamp, query, n_results)]
if "genre_counter" not in st.session_state:
    st.session_state.genre_counter = Counter()
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "reason_lang" not in st.session_state:
    st.session_state.reason_lang = "ar"


# ---------------- Theme injection (fake dark/light mode via CSS) ----------------
def inject_theme(dark: bool):
    if dark:
        css = """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color:#0e1117; color:#fafafa; }
        [data-testid="stSidebar"] { background-color:#161a23; }
        [data-testid="stSidebar"] * { color:#fafafa !important; }
        .stMarkdown, .stCaption, p, span, label, h1, h2, h3, h4 { color:#fafafa !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color:#fafafa !important; }
        div[data-testid="stTextInput"] input { background-color:#1e222b; color:#fafafa; }
        [data-testid="stVerticalBlockBorderWrapper"] { background-color:#1a1d26; border-color:#30333d !important; }

        /* --- Buttons: give them an explicit dark background so white text/icons stay visible --- */
        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"],
        [data-testid^="baseButton"] {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #41444e !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid^="baseButton"]:hover {
            background-color: #3a3d46 !important;
            border-color: #5a5e6b !important;
            color: #fafafa !important;
        }
        button[kind="primary"] {
            background-color: #e63946 !important;
            color: #ffffff !important;
            border: none !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color:#ffffff; color:#0e1117; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ---------------- Load resources ----------------
try:
    resources = load_resources()
    df, tfidf_vectorizer, tfidf_matrix, faiss_index, model, cols = resources
    load_ok = True
except Exception as e:
    load_ok = False
    resources = None
    st.error(f"Failed to load the required files. Make sure all the files exist in the same DATA_DIR path.\n\nDetails: {e}")


@st.cache_data(show_spinner=False)
def get_all_genres(_df, genre_col):
    if not genre_col:
        return []
    all_genres = set()
    for val in _df[genre_col].dropna():
        all_genres.update(as_list(val))
    return sorted(all_genres)


# ==========================================================
# SIDEBAR — navigation, theme toggle, filters
# ==========================================================
with st.sidebar:
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown("### 🎮 GameSense AI")
    with top_r:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(icon, help="Toggle dark / light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    
    st.caption("Smart Semantic Search & Recommendation")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Search", "⭐ For You", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ Search Settings**")
    alpha = st.slider("Semantic Search Weight", 0.0, 1.0, 0.6, 0.05)
    top_k = st.slider("Number of Results Shown", 3, 15, 5)

    genre_col = cols.get("genres") if load_ok else None
    all_genres = get_all_genres(df, genre_col) if load_ok else []
    selected_genres = st.multiselect("Filter by Genre", options=all_genres)

    st.markdown("---")
    st.markdown("**🕘 Search History**")
    if st.session_state.history:
        for q in reversed(st.session_state.history[-5:]):
            st.caption(f"• {q}")
    else:
        st.caption("No searches yet")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.history_log = []
            st.session_state.genre_counter = Counter()
            st.rerun()
    with c2:
        if st.session_state.history_log:
            hist_df = pd.DataFrame(st.session_state.history_log, columns=["time", "query", "results"])
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV", data=csv_bytes, file_name="search_history.csv", use_container_width=True)

inject_theme(st.session_state.dark_mode)
# ==========================================================
# Helper to render a single result card
# ==========================================================
def render_result_card(row, query, cols, key_prefix=""):
    info = explain_row(row, query, cols, lang=st.session_state.get("reason_lang", "ar"))
    title_col, desc_col = cols.get("title"), cols.get("description")
    title = row.get(title_col, "Unknown Title")
    with st.container(border=True):
        top_l, top_r = st.columns([6, 1])
        with top_l:
            st.markdown(f"### {title}")
        with top_r:
            fav_key = f"fav_{key_prefix}_{title}"
            is_fav = title in st.session_state.favorites
            if st.button("★" if is_fav else "☆", key=fav_key):
                if is_fav:
                    st.session_state.favorites.discard(title)
                else:
                    st.session_state.favorites.add(title)
                st.rerun()
        if desc_col:
            st.write(str(row.get(desc_col, ""))[:300])
        c1, c2, c3 = st.columns(3)
        c1.metric("Match Score", f"{info['match_score']}%")
        c2.metric("Semantic Similarity", f"{info['semantic_score']}%")
        c3.write(f"**Shared Genres:** {', '.join(info['shared_genres']) or '—'}")
        label = "💡 **Why recommended:**" if st.session_state.get("reason_lang", "ar") == "en" else "💡 **Why recommended:**"
        st.info(f"{label} {info['reason']}")


def apply_genre_filter(results, genre_col, selected):
    if not selected or not genre_col:
        return results
    mask = results[genre_col].apply(lambda v: any(g in as_list(v) for g in selected))
    return results[mask]
# ==========================================================
# PAGE: SEARCH
# ==========================================================
if page == "🔍 Search":
    st.title("🔍 Search for Your Next Game")
    if load_ok:
        query = st.text_input("A name, description, genre, or something like 'open world survival with crafting'")

        if st.button("Search", type="primary") and query.strip():
            st.session_state.history.append(query.strip())
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]

            results, has_meaningful_match = hybrid_search(query, resources, k=30, alpha=alpha)

            if not has_meaningful_match:
                st.session_state.last_results = results.iloc[0:0]  # empty
                st.session_state.last_query = query
                st.session_state.last_no_match = True
            else:
                results = apply_genre_filter(results, cols.get("genres"), selected_genres)
                results = personalize_rerank(results, st.session_state.history[:-1], cols)
                top_results = results.head(top_k)

                genre_col_ = cols.get("genres")
                for _, row in top_results.iterrows():
                    genres_list = as_list(row.get(genre_col_, "")) if genre_col_ else []
                    st.session_state.genre_counter.update(genres_list)

                st.session_state.history_log.append(
                    (dt.datetime.now().strftime("%Y-%m-%d %H:%M"), query.strip(), len(top_results))
                )
                st.session_state.last_results = top_results
                st.session_state.last_query = query
                st.session_state.last_no_match = False

        if st.session_state.get("last_no_match"):
            st.warning(
                "I couldn't find any game that's really close to what you're looking for. "
                "Try writing a clearer description or real words about the type of game (e.g., 'zombie survival crafting')."
            )
        elif st.session_state.last_results is not None and len(st.session_state.last_results) > 0:
            st.subheader(f"Top {len(st.session_state.last_results)} results")
            for i, (_, row) in enumerate(st.session_state.last_results.iterrows()):
                render_result_card(row, st.session_state.last_query, cols, key_prefix=f"search_{i}")
        elif st.session_state.last_query:
            st.warning("No matching results — try removing filters or changing your search term.")
    else:
        st.warning("The required files must be available first for search to work.")

# ==========================================================
# PAGE: RECOMMENDATIONS
# ==========================================================
elif page == "⭐ For You":
    st.title("⭐ Recommendations Based on Your Search")
    if load_ok and st.session_state.history:
        combined_query = " ".join(st.session_state.history[-HISTORY_LIMIT:])
        rec_results, has_match = hybrid_search(combined_query, resources, k=30, alpha=alpha)
        if has_match:
            rec_results = apply_genre_filter(rec_results, cols.get("genres"), selected_genres)
            rec_results = personalize_rerank(rec_results, st.session_state.history, cols)
            top_recs = rec_results.head(top_k)

            for i, (_, row) in enumerate(top_recs.iterrows()):
                render_result_card(row, combined_query, cols, key_prefix=f"rec_{i}")
        else:
            st.info("No clear recommendations today — try searching for games with a clearer description first.")
    else:
        st.info("Search for a few games first on the Search page, and you'll find recommendations here based on your interests.")

    if st.session_state.favorites:
        st.markdown("---")
        st.subheader("★ Your Favorite Games")
        for fav in st.session_state.favorites:
            st.write(f"• {fav}")

# ==========================================================
# PAGE: DASHBOARD
# ==========================================================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard — Search Analytics")

    if load_ok:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Games in Database", f"{len(df):,}")
        m2.metric("Number of Searches (Session)", len(st.session_state.history))
        m3.metric("Favorite Games", len(st.session_state.favorites))
        last_score = 0
        if st.session_state.last_results is not None and len(st.session_state.last_results) > 0:
            last_score = round(float(st.session_state.last_results["match_score"].max()) * 100, 1)
        m4.metric("Highest Match Score (Last Search)", f"{last_score}%")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Most Searched Genres**")
            if st.session_state.genre_counter:
                genres, counts = zip(*st.session_state.genre_counter.most_common(10))
                fig, ax = plt.subplots()
                ax.barh(genres, counts, color="#e63946")
                ax.invert_yaxis()
                ax.set_xlabel("Frequency")
                st.pyplot(fig)
            else:
                st.caption("Not enough data yet — try searching for some games.")

        with col_b:
            st.markdown("**Genre Distribution Across the Entire Database**")
            genre_col_ = cols.get("genres")
            if genre_col_:
                overall_counter = Counter()
                for val in df[genre_col_].dropna().head(2000):
                    overall_counter.update(as_list(val))
                if overall_counter:
                    genres2, counts2 = zip(*overall_counter.most_common(8))
                    fig2, ax2 = plt.subplots()
                    ax2.pie(counts2, labels=genres2, autopct="%1.0f%%")
                    st.pyplot(fig2)
            else:
                st.caption("No clear Genres column found in the data.")

        st.markdown("---")
        st.markdown("**Search History Log (with Timestamps)**")
        if st.session_state.history_log:
            log_df = pd.DataFrame(st.session_state.history_log, columns=["Time", "Query", "Result Count"])
            st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.caption("You haven't made any searches in this session yet.")
    else:
        st.warning("The required files must be available first for the statistics to show.")

# ==========================================================
# PAGE: ABOUT
# ==========================================================
else:
    st.title("ℹ️ About GameSense AI")
    st.write(
        "A smart search and recommendation system for games, combining semantic search "
        "(Sentence Transformers + FAISS) with text-based search (TF-IDF) in a single Hybrid system, "
        "ranking results based on your search history, and explaining the reasoning behind every "
        "recommendation with full transparency."
    )
    st.markdown("**Technologies Used:** Python, Pandas, NLTK, Scikit-learn, Sentence-Transformers, FAISS, Streamlit")