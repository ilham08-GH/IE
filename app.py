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
# 🔹 FUNGSI LOAD MODEL, TOKENIZER, DAN CBOW
# ==============================================================

@st.cache_resource
def load_assets():
    """Memuat model BiLSTM, tokenizer, dan CBOW Word2Vec"""
    base_dir = os.path.dirname(os.path.abspath(__file__))  # pastikan path aman

    model_path = os.path.join(base_dir, "ner_bilstm_cbow.keras")
    tokenizer_path = os.path.join(base_dir, "tokenizer_ner.pkl")
    cbow_path = os.path.join(base_dir, "cbow_embedding.model")

    # ✅ Debug info
    st.sidebar.write("📁 Current directory:", base_dir)
    st.sidebar.write("📂 Files:", os.listdir(base_dir))

    # Load model
    if not os.path.exists(model_path):
        st.stop()
        st.error(f"❌ File model tidak ditemukan di: {model_path}")
    model = load_model(model_path)

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        st.stop()
        st.error(f"❌ File tokenizer tidak ditemukan di: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load CBOW model
    if not os.path.exists(cbow_path):
        st.error(f"❌ File cbow_embedding.model tidak ditemukan di: {cbow_path}")
        w2v = None
    else:
        w2v = Word2Vec.load(cbow_path)

    return model, tokenizer, w2v


model, tokenizer, w2v = load_assets()

# ==============================================================
# 🔹 KONFIGURASI DASAR
# ==============================================================

MAX_LEN = 100
EMBED_DIM = 10  # harus sama dengan embedding_dim model kamu
label_map = {
    0: "O",
    1: "PERSON",
    2: "LAW",
    3: "LOCATION",
    4: "ACTION"
}

# ==============================================================
# 🔹 PREPROCESS TEKS DENGAN CBOW
# ==============================================================

def preprocess_text(text):
    """
    Mengubah teks menjadi urutan embedding vektor menggunakan CBOW Word2Vec.
    Hasil shape: (1, MAX_LEN, EMBED_DIM)
    """
    text = text.lower().strip()
    words = text.split()
    seq = []

    for w in words:
        if w2v and w in w2v.wv:
            seq.append(w2v.wv[w])
        else:
            seq.append(np.zeros(EMBED_DIM))  # kata tak dikenal

    # Padding / truncate agar panjang tetap MAX_LEN
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
        preds = model.predict(seq)[0]  # hasil shape (MAX_LEN, n_labels)
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
    Model **BiLSTM + CBOW** untuk mendeteksi entitas dalam teks hukum berbahasa Indonesia 🇮🇩  
    Pastikan file berikut ada di folder yang sama:
    - `ner_bilstm_cbow.keras`
    - `tokenizer_ner.pkl`
    - `cbow_embedding.model`
    """
)

st.sidebar.subheader("🧠 Info Model")
st.sidebar.write("Input shape:", model.input_shape)
st.sidebar.write("Embedding dimensi:", EMBED_DIM)

text_input = st.text_area(
    "Masukkan teks hukum di sini:",
    "Terdakwa Andi melanggar Pasal 362 KUHP tentang pencurian di Jakarta."
)

if st.button("🔍 Prediksi Entitas"):
    with st.spinner("Memproses teks..."):
        entities = predict_entities(text_input)
        st.subheader("📊 Hasil Prediksi:")

        colors = {
            "PERSON": "#B3E5FC",
            "LAW": "#C8E6C9",
            "LOCATION": "#FFF9C4",
            "ACTION": "#F8BBD0"
        }

        html_text = ""
        for token, label in entities:
            if label != "O":
                color = colors.get(label, "#E0E0E0")
                html_text += f"<span style='background-color:{color}; padding:2px 5px; border-radius:4px;'>{token}</span> "
            else:
                html_text += f"{token} "
        st.markdown(html_text, unsafe_allow_html=True)

st.markdown("---")
st.caption("🧠 Model: BiLSTM + CBOW | Dibuat untuk analisis teks hukum Indonesia 🇮🇩")
