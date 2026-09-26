import os
import re
import random
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MedBot — Medical Q&A Chatbot", page_icon="🏥", layout="centered")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Keep this modest for free hosting (Streamlit Community Cloud has ~1GB RAM).
# Set the MEDBOT_SAMPLE_SIZE env var (Settings -> Secrets/Advanced) to change it,
# or set it to 0 to use the full ~16k-row dataset.
SAMPLE_SIZE = int(os.environ.get("MEDBOT_SAMPLE_SIZE", "3000"))

# Set MEDBOT_USE_SBERT=0 to force lightweight TF-IDF-only mode (no SBERT/torch
# download at all) — useful if the host runs out of memory loading SBERT.
USE_SBERT_REQUESTED = os.environ.get("MEDBOT_USE_SBERT", "1") != "0"

CSV_CANDIDATES = [
    "medquad.csv",
    os.path.join(os.path.dirname(__file__), "medquad.csv"),
    "/content/medquad.csv",
    "/mnt/user-data/uploads/medquad.csv",
]

# ---------------------------------------------------------------------------
# NLTK setup (downloaded once per app instance, then cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan NLTK...")
def setup_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


setup_nltk()

from nltk.tokenize import word_tokenize  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

# ---------------------------------------------------------------------------
# Optional SBERT (falls back to TF-IDF if unavailable / disabled / too heavy)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model SBERT (hanya sekali)...")
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


SBERT_MODEL = load_sbert() if USE_SBERT_REQUESTED else None
USE_SBERT = SBERT_MODEL is not None

# ---------------------------------------------------------------------------
# Preprocessing (same rules as the notebook)
# ---------------------------------------------------------------------------
INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "atau", "juga", "dalam", "tidak", "akan", "ada",
    "saya", "kamu", "anda", "ia", "mereka", "kami", "kita", "bisa",
    "sudah", "bila", "jika", "maka", "oleh", "karena", "apa",
    "bagaimana", "berapa", "kapan", "dimana", "siapa", "apakah", "cara",
    "lebih", "sangat", "dapat", "nya", "pun", "lagi", "belum",
    "telah", "namun", "tapi", "serta", "meski", "agar", "supaya", "hal",
    "the", "is", "are", "was", "what", "how", "why", "when", "where",
}


@st.cache_resource
def get_stopwords_and_lemmatizer():
    english_stopwords = set(stopwords.words("english"))
    return INDONESIAN_STOPWORDS | english_stopwords, WordNetLemmatizer()


ALL_STOPWORDS, LEMMATIZER = get_stopwords_and_lemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset loading (medquad.csv -> category / question / answer)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat & memproses dataset...")
def load_dataset(sample_size):
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        st.error(
            "medquad.csv tidak ditemukan. Taruh file ini di folder yang sama dengan app.py "
            "(atau di root repo untuk deploy ke Streamlit Community Cloud)."
        )
        st.stop()

    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["question", "answer"]).copy()
    raw["focus_area"] = raw["focus_area"].fillna("Umum")

    if sample_size and len(raw) > sample_size:
        raw = raw.sample(n=sample_size, random_state=42).reset_index(drop=True)

    data = raw.rename(columns={"focus_area": "category"})[["category", "question", "answer"]]
    data = data.reset_index(drop=True)
    data["processed_question"] = data["question"].apply(preprocess_text)
    return data


df = load_dataset(SAMPLE_SIZE)


# ---------------------------------------------------------------------------
# Chatbot engine (same logic as MedicalChatbotEngineV3 in the notebook)
# ---------------------------------------------------------------------------
class MedicalChatbotEngineV3:
    def __init__(self, dataframe, threshold=0.5, top_k=3):
        self.df = dataframe
        self.threshold = threshold if USE_SBERT else 0.5
        self.top_k = top_k
        self.conversation_history = []
        self._build_index()
        self._define_rules()

    def _build_index(self):
        if USE_SBERT:
            self.sbert_embeddings = SBERT_MODEL.encode(
                self.df["question"].tolist(), convert_to_tensor=True
            )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_question"])

    def _define_rules(self):
        self.rules = {
            "emergency": {
                "patterns": [
                    r"(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan|"
                    r"chest pain|can'?t breathe|unconscious|severe bleeding)"
                ],
                "responses": [
                    "🚨 DARURAT! Segera hubungi layanan gawat darurat setempat "
                    "(mis. 119/112 di Indonesia, 911 di AS) atau ke IGD terdekat!"
                ],
            },
            "greeting": {
                "patterns": [r"\b(halo|hai|hi|hello|hey)\b"],
                "responses": ["👋 Halo! Ada pertanyaan kesehatan yang bisa saya bantu?"],
            },
        }

    def _check_rules(self, text):
        for _intent, data in self.rules.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text.lower()):
                    return random.choice(data["responses"])
        return None

    def _search_sbert(self, query):
        from sentence_transformers import util
        emb = SBERT_MODEL.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top_results = np.argsort(-scores)[: self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _search_tfidf(self, query):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][: self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def _build_context_query(self, user_input):
        if self.conversation_history:
            return self.conversation_history[-1] + " " + user_input
        return user_input

    def get_response(self, user_input):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        query = self._build_context_query(user_input)
        results = self._search_sbert(query) if USE_SBERT else self._search_tfidf(query)
        method = "TF-IDF"

        best_idx, best_score = results[0]
        if best_score < self.threshold:
            self.conversation_history.append(user_input)
            return "🤔 Tidak menemukan jawaban yang cukup relevan. Coba ungkapkan pertanyaan dengan kata lain."

        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]["question"]
            bonus = sum(1 for word in user_input.lower().split() if word in text.lower())
            boosted.append((idx, score + 0.05 * bonus))
        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]

        row = self.df.iloc[best_idx]
        self.conversation_history.append(user_input)

        return (
            f"**Kategori:** {row['category']}  ·  _{method}_\n\n"
            f"{row['answer']}\n\n"
            f"---\n⚠️ Info edukasi, bukan pengganti diagnosis dokter."
        )


@st.cache_resource(show_spinner="Membangun index chatbot (hanya sekali)...")
def build_bot(_df):
    return MedicalChatbotEngineV3(_df)


bot = build_bot(df)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏥 MedBot — Medical Q&A Chatbot")
st.caption(
    f"Dataset: MedQuAD · {len(df)} pasangan Q&A · {df['category'].nunique()} kategori · "
)
st.info(
    "⚠️ Chatbot ini untuk edukasi saja dan bukan pengganti konsultasi dengan dokter. "
    "Untuk kondisi darurat, segera hubungi layanan gawat darurat setempat."
)

with st.sidebar:
    st.header("Topik populer")
    top_categories = df["category"].value_counts().head(8).index.tolist()
    for cat in top_categories:
        if st.button(cat, use_container_width=True):
            st.session_state["quick_query"] = f"What is {cat}?"
            st.rerun()
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        bot.conversation_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ketik pertanyaan kesehatan...")
if "quick_query" in st.session_state:
    user_input = st.session_state.pop("quick_query")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            response = bot.get_response(user_input)
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
