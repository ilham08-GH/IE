import streamlit as st
import numpy as np
import re
import joblib
from gensim.models import Word2Vec
from keras.models import load_model
from tensorflow.keras.utils import pad_sequences

# --- (BARU) PETA NAMA TAG ---
# Peta untuk menerjemahkan tag internal (DEFN) ke nama yang mudah dibaca
TAG_MAP = {
    'ADVO': 'Advokat/Pengacara',
    'ARTV': 'Pasal/Artikel Hukum',
    'CRIA': 'Tindak Pidana',
    'DEFN': 'Identitas Terdakwa/Tergugat',
    'JUDG': 'Hakim',
    'JUDP': 'Institusi Pengadilan',
    'PENA': 'Jaksa Penuntut Umum', # Asumsi dari 'Penuntut'
    'PROS': 'Jaksa Penuntut Umum', # Asumsi dari 'Prosecutor'
    'PUNI': 'Hukuman/Pidana',
    'REGI': 'Nomor Registrasi',
    'TIMV': 'Durasi Waktu',
    'VERN': 'Vonis/Putusan',
    # Tambahkan tag lain jika ada
}


# --- FUNGSI HELPER UNTUK EMBEDDING ---
def get_word_embedding(word, model_wv, vector_size):
    if word in model_wv:
        return model_wv[word]
    else:
        return np.zeros(vector_size)

# --- FUNGSI UNTUK MEMUAT SEMUA MODEL ---
@st.cache_resource
def load_all_models():
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
        st.error("File model tidak ditemukan! Pastikan 'model_bilstm_pidana.h5', 'cbow_pidana.model', dan 'app_data.pkl' ada di folder yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

# --- FUNGSI PREDIKSI TAG (Tidak Berubah) ---
def predict_sentence_tags(raw_text, assets):
    try:
        keras_model = assets['model']
        cbow_model = assets['cbow_model']
        tag_map = assets['idx2tag']
        max_len = assets['MAX_LEN']
        embedding_size = assets['vector_size']
        
        cleaned_text = raw_text.lower()
        cleaned_text = re.sub(r'[\d]', 'X', cleaned_text)
        
        words = re.findall(r"[\w']+|[.,!?;:]", cleaned_text)
        
        if not words:
            return "Input kosong. Silakan masukkan kalimat."
            
        X_new = [[get_word_embedding(w, cbow_model.wv, embedding_size) for w in words]]
        
        X_new = pad_sequences(maxlen=max_len, 
                              sequences=X_new, 
                              padding="post", 
                              dtype='float32', 
                              value=np.zeros(embedding_size))
        
        p = keras_model.predict(X_new)
        p = np.argmax(p, axis=-1)
        
        results = []
        for i in range(len(words)):
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

# --- (FUNGSI BARU) EKSTRAKSI ENTITAS ---
def extract_entities(predictions):
    """
    Mengambil daftar (kata, tag) dan merakitnya menjadi entitas yang utuh.
    Contoh: (terdakwa, B_DEFN), (budi, I_DEFN) -> DEFN: ["terdakwa budi"]
    """
    entities = {}
    current_entity_words = []
    current_entity_tag = None

    for word, tag in predictions:
        if tag.startswith('B_'):
            # Jika ada entitas sebelumnya, simpan
            if current_entity_tag:
                entity_text = " ".join(current_entity_words)
                if current_entity_tag not in entities:
                    entities[current_entity_tag] = []
                entities[current_entity_tag].append(entity_text)
            
            # Mulai entitas baru
            current_entity_words = [word]
            current_entity_tag = tag.split('_', 1)[1] # Ambil 'DEFN' dari 'B_DEFN'
        
        elif tag.startswith('I_'):
            # Lanjutkan entitas jika tag-nya cocok
            if current_entity_tag and tag.split('_', 1)[1] == current_entity_tag:
                current_entity_words.append(word)
            else:
                # Jika tag 'I_' tidak cocok (atau muncul tanpa 'B_'), abaikan
                pass
        
        else: # (Tag 'O' atau lainnya)
            # Jika ada entitas yang sedang dibangun, simpan
            if current_entity_tag:
                entity_text = " ".join(current_entity_words)
                if current_entity_tag not in entities:
                    entities[current_entity_tag] = []
                entities[current_entity_tag].append(entity_text)
            
            # Reset
            current_entity_words = []
            current_entity_tag = None

    # Simpan entitas terakhir setelah loop selesai
    if current_entity_tag:
        entity_text = " ".join(current_entity_words)
        if current_entity_tag not in entities:
            entities[current_entity_tag] = []
        entities[current_entity_tag].append(entity_text)

    return entities


# --- (MODIFIKASI) ANTARMUKA APLIKASI STREAMLIT ---

assets = load_all_models()

st.title("⚖️ Aplikasi Ekstraksi Informasi Teks Hukum")
st.subheader("Dibangun dengan Bi-LSTM dan Streamlit")
st.markdown("Aplikasi ini melakukan **Ekstraksi Informasi** (Tahap 2) dengan cara memprediksi *tag* entitas (Tahap 1) dan kemudian merakitnya menjadi informasi yang utuh.")

if assets: 
    input_text = st.text_area(
        "Masukkan kalimat teks hukum di sini:",
        "Bahwa terdakwa Budi Hartono alias Bodong bin Sukiman, didampingi oleh Penasihat Hukumnya, Dr. Sinar Pagi, S.H., M.H., telah terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana pencurian dengan pemberatan sebagaimana diatur dalam Pasal 363 ayat (1) KUHP, dan oleh karenanya Majelis Hakim menjatuhkan pidana penjara selama 1 (satu) tahun dan 6 (enam) bulan.",
        height=150
    )

    if st.button("Ekstrak Informasi"):
        if input_text:
            with st.spinner("Sedang memprediksi tag..."):
                # TAHAP 1: PREDIKSI TAG
                predictions = predict_sentence_tags(input_text, assets)
            
            if predictions:
                with st.spinner("Sedang merakit entitas..."):
                    # TAHAP 2: EKSTRAKSI ENTITAS
                    entities = extract_entities(predictions)
                
                st.success("Ekstraksi Selesai!")
                
                # --- (BARU) TAMPILKAN HASIL EKSTRAKSI ---
                st.subheader("Informasi Entitas yang Diekstraksi:")
                
                if not entities:
                    st.warning("Tidak ada entitas yang ditemukan.")
                else:
                    # Tampilkan entitas yang diekstrak dengan rapi
                    for entity_type, entity_list in entities.items():
                        # Gunakan nama yang mudah dibaca dari TAG_MAP
                        friendly_name = TAG_MAP.get(entity_type, entity_type)
                        
                        st.markdown(f"**{friendly_name}:**")
                        for entity in entity_list:
                            # Tampilkan hasil ekstraksi dalam kotak info
                            st.info(f"{entity}")
                
                # --- TAMPILKAN HASIL TAGGING MENTAH DI DALAM EXPAΝDER ---
                with st.expander("Lihat Rincian Tag per Kata (Hasil Tahap 1)"):
                    output_md = "| Kata | Tag |\n| :--- | :--- |\n"
                    for word, tag in predictions:
                        output_md += f"| {word} | **{tag}** |\n"
                    st.markdown(output_md)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")
