import streamlit as st
import numpy as np
import re  # Pastikan 're' (Regular Expressions) diimpor
import joblib
from gensim.models import Word2Vec
from keras.models import load_model
from tensorflow.keras.utils import pad_sequences

# --- PENGATURAN HALAMAN APLIKASI ---
st.set_page_config(
    page_title="Prediksi Tag Teks Hukum",
    page_icon="⚖️",
    layout="centered"
)

# --- FUNGSI HELPER UNTUK EMBEDDING ---
def get_word_embedding(word, model_wv, vector_size):
    """
    Helper untuk mendapatkan embedding dari cbow_model.
    Jika kata tidak ditemukan, kembalikan vektor nol.
    """
    if word in model_wv:
        return model_wv[word]
    else:
        return np.zeros(vector_size)

# --- FUNGSI UNTUK MEMUAT SEMUA MODEL ---
@st.cache_resource
def load_all_models():
    """
    Memuat semua model dan data yang disimpan dari disk.
    Menggunakan cache Streamlit agar hanya dimuat sekali.
    """
    try:
        # 1. Muat model Keras
        model = load_model('model_bilstm_pidana.h5')
        
        # 2. Muat model Word2Vec (gensim)
        cbow_model = Word2Vec.load('cbow_pidana.model')
        
        # 3. Muat data pendukung (kamus tag, dll)
        app_data = joblib.load('app_data.pkl')
        
        # Kembalikan semua dalam satu dictionary
        return {
            "model": model,
            "cbow_model": cbow_model,
            "idx2tag": app_data['idx2tag'],
            "MAX_LEN": app_data['MAX_LEN'],
            "vector_size": app_data['vector_size']
        }
    except FileNotFoundError:
        st.error("File model tidak ditemukan! Pastikan 'model_bilstm_pidana.h5', 'cbow_pidana.model', dan 'app_data.pkl' ada di folder yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

# --- FUNGSI PREDIKSI UTAMA (YANG SUDAH DIMODIFIKASI) ---
def predict_sentence_tags(raw_text, assets):
    """
    Fungsi lengkap untuk memproses teks mentah menjadi prediksi.
    (VERSI DIPERBARUI DENGAN TOKENISASI YANG LEBIH BAIK)
    """
    try:
        # 1. Unpack semua model dan data dari 'assets'
        keras_model = assets['model']
        cbow_model = assets['cbow_model']
        tag_map = assets['idx2tag']
        max_len = assets['MAX_LEN']
        embedding_size = assets['vector_size']
        
        # 2. Preprocessing Teks
        cleaned_text = raw_text.lower()
        cleaned_text = re.sub(r'[\d]', 'X', cleaned_text) # Mengganti semua digit dengan 'X'
        
        # --- PERBAIKAN UTAMA ADA DI SINI ---
        # Gunakan regex untuk memisahkan kata DAN tanda baca
        # Ini akan mengubah "sukiman," menjadi ["sukiman", ","]
        words = re.findall(r"[\w']+|[.,!?;:]", cleaned_text)
        # ------------------------------------
        
        if not words:
            return "Input kosong. Silakan masukkan kalimat."
            
        # 3. Konversi Kata menjadi Embedding (Word2Vec)
        X_new = [[get_word_embedding(w, cbow_model.wv, embedding_size) for w in words]]
        
        # 4. Lakukan Padding
        X_new = pad_sequences(maxlen=max_len, 
                              sequences=X_new, 
                              padding="post", 
                              dtype='float32', 
                              value=np.zeros(embedding_size))
        
        # 5. Lakukan Prediksi dengan Model Keras
        p = keras_model.predict(X_new)
        p = np.argmax(p, axis=-1)
        
        # 6. Format Output (Ubah indeks tag kembali ke teks)
        results = []
        for i in range(len(words)):
            # Perlakuan khusus: Jika token adalah tanda baca, tag-nya 'O'
            if words[i] in [",", ".", ":", ";", "!", "?"]:
                tag = "O"
            else:
                tag_index = p[0][i]
                tag = tag_map[tag_index]
            
            results.append((words[i], tag))
            
        return results

    except Exception as e:
        st.error(f"Terjadi Error saat prediksi: {str(e)}")
        return None

# --- MEMBANGUN ANTARMUKA APLIKASI STREAMLIT ---

# 1. Muat semua model saat aplikasi dimulai
assets = load_all_models()

# 2. Tampilkan Judul dan Subjudul
st.title("⚖️ Aplikasi Prediksi Tag Entitas Hukum")
st.subheader("Dibangun dengan Bi-LSTM dan Streamlit")
st.markdown("Aplikasi ini dapat mengenali entitas dalam teks hukum (misal: terdakwa, pasal, hukuman) berdasarkan model yang dilatih pada 200 putusan pidana.")

# 3. Hanya tampilkan UI jika model berhasil dimuat
if assets: 
    # 4. Buat area input teks
    input_text = st.text_area(
        "Masukkan kalimat teks hukum di sini:",
        "Bahwa terdakwa Budi Hartono alias Bodong bin Sukiman, didampingi oleh Penasihat Hukumnya, Dr. Sinar Pagi, S.H., M.H., telah terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana pencurian dengan pemberatan sebagaimana diatur dalam Pasal 363 ayat (1) KUHP, dan oleh karenanya Majelis Hakim menjatuhkan pidana penjara selama 1 (satu) tahun dan 6 (enam) bulan.",
        height=150
    )

    # 5. Buat tombol untuk memulai prediksi
    if st.button("Prediksi Tag"):
        if input_text:
            # Tampilkan spinner selagi model bekerja
            with st.spinner("Sedang memprediksi..."):
                predictions = predict_sentence_tags(input_text, assets)
            
            # 6. Tampilkan hasil jika prediksi berhasil
            if predictions:
                st.success("Prediksi Selesai!")
                
                # Format hasil sebagai tabel markdown
                output_md = "**Hasil Prediksi:**\n\n| Kata | Tag |\n| :--- | :--- |\n"
                for word, tag in predictions:
                    # Tambahkan bold pada tag agar mudah dibaca
                    output_md += f"| {word} | **{tag}** |\n"
                
                st.markdown(output_md)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")
