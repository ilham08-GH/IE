# ==============================================================
# 🔎 Named Entity Recognition (NER) Hukum — BiLSTM + CBOW
# Dibuat agar siap digunakan di Streamlit
# ==============================================================
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================
# 🔹 LOAD MODEL & TOKENIZER
# ==============================================================

@st.cache_resource
def load_assets():
    """Memuat model BiLSTM dan tokenizer"""
    model = load_model("ner_bilstm_cbow.keras")
    with open("tokenizer_ner.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# ==============================================================
# 🔹 KONFIGURASI DASAR
# ==============================================================

MAX_LEN = 100  # harus sama seperti saat training
label_map = {
    0: "O",
    1: "PERSON",
    2: "LAW",
    3: "LOCATION",
    4: "ACTION"
}

# ==============================================================
# 🔹 PREPROCESS TEKS
# ==============================================================

def preprocess_text(text: str):
    """Membersihkan dan mengubah teks menjadi urutan numerik"""
    text = text.lower().strip()
    seq = tokenizer.texts_to_sequences([text])  # ❌ jangan pakai text.split()
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

# ==============================================================
# 🔹 FUNGSI PREDIKSI ENTITAS
# ==============================================================

def predict_entities(text: str):
    """Melakukan prediksi entitas hukum dari teks input"""
    if not text.strip():
        return [("⚠️", "Input kosong")]

    try:
        seq = preprocess_text(text)
        preds = model.predict(seq)[0]  # (MAX_LEN, num_labels)

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

st.set_page_config(page_title="NER Hukum BiLSTM", page_icon="⚖️", layout="centered")

st.title("⚖️ Named Entity Recognition (NER) Hukum Indonesia")
st.markdown(
    """
    Model **BiLSTM + CBOW** untuk mendeteksi entitas dalam teks hukum berbahasa Indonesia 🇮🇩  
    Jalankan dengan mengetik teks hukum di bawah, lalu klik tombol **Prediksi Entitas**.
    """
)

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
st.caption("🧠 Model: BiLSTM + CBOW | Dibuat untuk analisis teks hukum Indonesia")
