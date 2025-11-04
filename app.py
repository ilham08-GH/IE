import streamlit as st
import numpy as np
import re
import joblib
from gensim.models import Word2Vec
from keras.models import load_model
from tensorflow.keras.utils import pad_sequences

# --- PENGATURAN HALAMAN (Opsional) ---
st.set_page_config(
    page_title="Prediksi Tag Teks Hukum",
    page_icon="⚖️",
    layout="centered"
)

# --- FUNGSI HELPER (Sama seperti sebelumnya) ---

def get_word_embedding(word, model_wv, vector_size):
    """
    Helper untuk mendapatkan embedding dari cbow_model.
    """
    if word in model_wv:
        return model_wv[word]
    else:
        return np.zeros(vector_size)

# --- FUNGSI UNTUK MEMUAT MODEL (PENTING!) ---
# Menggunakan cache agar model hanya di-load sekali
@st.cache_resource
def load_all_models():
    """
    Memuat semua model dan data yang disimpan dari disk.
    """
    try:
        model = load_model('model_bilstm_pidana.h5')
        cbow_model = Word2Vec.load('cbow_pidana.model')
        app_data = joblib.load('app_data.pkl')
        
        return {
            "model": model,
            "cbow_model": cbow_model,
            "idx2tag": app_data['idx2tag'],
            "MAX_LEN": app_data['MAX_LEN'],
            "vector_size": app_data['vector_size']
        }
    except FileNotFoundError:
        st.error("File model tidak ditemukan! Pastikan 'model_bilstm_pidana.h5', 'cbow_pidana.model', dan 'app_data.pkl' ada di folder yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

# --- FUNGSI PREDIKSI UTAMA ---
def predict_sentence_tags(raw_text, assets):
    """
    Fungsi lengkap untuk memproses teks mentah menjadi prediksi.
    """
    try:
        # Unpack assets
        keras_model = assets['model']
        cbow_model = assets['cbow_model']
        tag_map = assets['idx2tag']
        max_len = assets['MAX_LEN']
        embedding_size = assets['vector_size']
        
        # 1. Preprocessing
        cleaned_text = raw_text.lower()
        cleaned_text = re.sub(r'[\d]', 'X', cleaned_text) # Mengganti digit
        words = cleaned_text.split()
        
        if not words:
            return "Input kosong. Silakan masukkan kalimat."
            
        # 2. Konversi ke Embedding
        X_new = [[get_word_embedding(w, cbow_model.wv, embedding_size) for w in words]]
        
        # 3. Padding
        X_new = pad_sequences(maxlen=max_len, 
                              sequences=X_new, 
                              padding="post", 
                              dtype='float32', 
                              value=np.zeros(embedding_size))
        
        # 4. Prediksi
        p = keras_model.predict(X_new)
        p = np.argmax(p, axis=-1)
        
        # 5. Format Output (dalam bentuk list)
        results = []
        for i in range(len(words)):
            tag_index = p[0][i]
            tag = tag_map[tag_index]
            results.append((words[i], tag))
            
        return results

    except Exception as e:
        st.error(f"Terjadi Error saat prediksi: {str(e)}")
        return None

# --- MEMBANGUN ANTARMUKA APLIKASI STREAMLIT ---

# 1. Muat semua model
assets = load_all_models()

# Judul Aplikasi
st.title("⚖️ Aplikasi Prediksi Tag Entitas Hukum")
st.subheader("Dibangun dengan Bi-LSTM dan Streamlit")

# Area input teks
if assets: # Hanya tampilkan jika model berhasil dimuat
    input_text = st.text_area(
        "Masukkan kalimat teks hukum di sini:",
        "contoh: terdakwa telah melakukan pencurian uang sebesar 100000 rupiah",
        height=100
    )

    # Tombol untuk prediksi
    if st.button("Prediksi Tag"):
        if input_text:
            with st.spinner("Sedang memprediksi..."):
                predictions = predict_sentence_tags(input_text, assets)
            
            if predictions:
                st.success("Prediksi Selesai!")
                
                # Tampilkan hasil dalam format yang bagus
                # Kita bisa pakai markdown untuk membuat tabel
                output_md = "**Hasil Prediksi:**\n\n| Kata | Tag |\n| :--- | :--- |\n"
                for word, tag in predictions:
                    output_md += f"| {word} | **{tag}** |\n"
                
                st.markdown(output_md)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")