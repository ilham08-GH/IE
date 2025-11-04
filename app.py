# ==============================================================
# ⚖️ Named Entity Recognition (NER) Hukum — BiLSTM + CBOW
# ==============================================================
import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec

# ==============================================================
# 🔹 FUNGSI LOAD MODEL, TOKENIZER, CBOW, DAN LABEL MAP
# ==============================================================

@st.cache_resource
def load_assets():
    """Memuat model BiLSTM, tokenizer, CBOW Word2Vec, dan label_map"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "ner_bilstm_cbow.keras")
    tokenizer_path = os.path.join(base_dir, "tokenizer_ner.pkl")
    cbow_path = os.path.join(base_dir, "cbow_embedding.model")
    labelmap_path = os.path.join(base_dir, "label_map.pkl")

    # ✅ Debug info sidebar
    st.sidebar.write("📁 Current directory:", base_dir)
    st.sidebar.write("📂 Files:", os.listdir(base_dir))

    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan di: {model_path}")
        st.stop()
    model = load_model(model_path)

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ File tokenizer tidak ditemukan di: {tokenizer_path}")
        st.stop()
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load CBOW
    if not os.path.exists(cbow_path):
        st.error(f"❌ File cbow_embedding.model tidak ditemukan di: {cbow_path}")
        w2v = None
    else:
        w2v = Word2Vec.load(cbow_path)

    # Load label_map
    if not os.path.exists(labelmap_path):
        st.warning("⚠️ label_map.pkl tidak ditemukan. Menggunakan default mapping.")
        label_map = {0: "O"}
    else:
        with open(labelmap_path, "rb") as f:
            label_map = pickle.load(f)

    return model, tokenizer, w2v, label_map


model, tokenizer, w2v, label_map = load_assets()

# ==============================================================
# 🔹 KONFIGURASI DASAR
# ==============================================================

MAX_LEN = 100
EMBED_DIM = 10

# ==============================================================
# 🔹 PREPROCESS TEKS DENGAN CBOW
# ==============================================================

def preprocess_text(text):
    """Mengubah teks menjadi urutan embedding vektor CBOW Word2Vec."""
    text = text.lower().strip()
    words = text.split()
    seq = []

    for w in words:
        if w2v and w in w2v.wv:
            seq.append(w2v.wv[w])
        else:
            seq.append(np.zeros(EMBED_DIM))  # kata tak dikenal

    if len(seq) < MAX_LEN:
        seq += [np.zeros(EMBED_DIM)] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]

    return np.array([seq])  # shape (1, MAX_LEN, EMBED_DIM)

# ==============================================================
# 🔹 PREDIKSI ENTITAS
# ==============================================================

def predict_entities(text):
    if not text.strip():
        return [("⚠️", "Input kosong")]

    try:
        seq = preprocess_text(text)
        preds = model.predict(seq)[0]
        tokens = text.split()
        results = []

        for i, token in enumerate(tokens[:MAX_LEN]):
            ent_label = label_map.get(np.argmax(preds[i]), "O")
            results.append((token, ent_label))
        return results

    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi: {e}")
        return []

# ==============================================================
# 🔹 STREAMLIT UI
# ==============================================================

st.set_page_config(page_title="NER Hukum BiLSTM+CBOW", page_icon="⚖️", layout="centered")

st.title("⚖️ Named Entity Recognition (NER) Hukum Indonesia")
st.markdown(
    """
    Model **BiLSTM + CBOW** untuk mendeteksi entitas dalam teks hukum 🇮🇩  
    Pastikan file berikut ada di folder yang sama:
    - `ner_bilstm_cbow.keras`
    - `tokenizer_ner.pkl`
    - `cbow_embedding.model`
    - `label_map.pkl`
    """
)

st.sidebar.subheader("🧠 Info Model")
st.sidebar.write("Input shape:", model.input_shape)
st.sidebar.write("Embedding dimensi:", EMBED_DIM)
st.sidebar.write("Jumlah label:", len(label_map))

text_input = st.text_area(
    "Masukkan teks hukum di sini:",
    "Terdakwa Andi melanggar Pasal 362 KUHP tentang pencurian di Jakarta."
)

if st.button("🔍 Prediksi Entitas"):
    with st.spinner("Memproses teks..."):
        entities = predict_entities(text_input)
        st.subheader("📊 Hasil Prediksi:")

        # warna untuk tiap entitas
        colors = {
            "PER": "#B3E5FC", "ORG": "#D1C4E9", "LOC": "#FFF9C4",
            "LAW": "#C8E6C9", "ACT": "#F8BBD0", "DATE": "#FFECB3",
            "NUM": "#B2DFDB", "PENALTY": "#FFCDD2", "ARTICLE": "#DCEDC8",
            "EVENT": "#E1BEE7", "MISC": "#CFD8DC"
        }

        html_text = ""
        for token, label in entities:
            if label != "O":
                ent_type = label.split("-")[-1]
                color = colors.get(ent_type, "#E0E0E0")
                html_text += f"<span style='background-color:{color}; padding:2px 5px; border-radius:4px;'>{token} <sub>({label})</sub></span> "
            else:
                html_text += f"{token} "
        st.markdown(html_text, unsafe_allow_html=True)

st.markdown("---")
st.caption("🧠 Model: BiLSTM + CBOW | Dibuat untuk analisis teks hukum Indonesia 🇮🇩")
