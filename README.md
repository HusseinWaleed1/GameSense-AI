# 🎮 GameSense AI
### Smart Semantic Search & Personalized Recommendation System

---

## 📖 About the Project

**GameSense AI** is an intelligent search and recommendation system built for game discovery. Instead of relying on rigid keyword matching, it *understands* what a user is looking for — even when they describe a game in their own words, like:

> "open world survival with crafting"

The system combines **semantic understanding**, **classic text search**, and **personalization** to surface the most relevant games, then explains *why* each one was recommended — turning a black-box recommendation engine into a transparent, trustworthy experience.

## 💡 Why It Matters

Traditional search on most game platforms depends on exact keyword matches — if you don't know the exact title, you're stuck. GameSense AI solves this by:

- **Understanding meaning, not just words** — powered by sentence embeddings, so vague or descriptive queries still return great results.
- **Learning from behavior** — recommendations get sharper the more a user searches, based on their real interests.
- **Explaining every result** — no black box. Every recommendation comes with a clear reason.
- **Balancing precision and recall** — a hybrid approach blends semantic and keyword-based search so neither weak phrasing nor rare terminology gets lost.

## ⚙️ How It Works

The system is built around a **Hybrid Search Pipeline**:

1. **TF-IDF Search** — classic keyword-based matching, great at catching exact terms and rare vocabulary.
2. **Semantic Search (Sentence Transformers + FAISS)** — embeds the query and the game corpus into vector space to capture *meaning*, not just words.
3. **Hybrid Scoring** — the two signals are normalized and blended (`alpha * semantic + (1 - alpha) * tfidf`) into a single match score, with a quality gate to avoid confidently returning weak matches.
4. **Personalized Re-ranking** — results are boosted based on overlap with the user's search history (shared genres/tags), so recommendations get more tailored over time.
5. **Explainability Layer** — every result is paired with a human-readable reason (e.g. shared genre, semantic closeness, past search similarity).
6. **Interactive Interface (Streamlit)** — a clean, full-featured UI with search, personalized recommendations, an analytics dashboard, favorites, dark mode, and downloadable search history.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NLTK` · `Scikit-learn` · `Sentence-Transformers` · `FAISS` · `Streamlit`

## 🚀 Running the App

```bash
streamlit run streamlit_app.py
```

Make sure the following pre-computed artifacts are available in the project directory:
- `final_df.pkl`
- `tfidf_vectorizer.pkl`
- `tfidf_matrix.pkl`
- `game_index.faiss`
- `sentence_transformer_model/`

---

## 👥 Team Members

| Name |
|------|
| Hussein Waleed Hussein |
| George Essam Saber |
| Malak Hassan Ali |
| Esraa Ahmed Abdel Moneim |
| Mohamed Adel Labeeb |

---

<p align="center"><i>Built with ❤️ to make finding your next favorite game effortless.</i></p>
