import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tomat", 
    page_icon="🍅", 
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size:42px !important; font-weight: bold; color: #E74C3C; text-align: center; margin-bottom: 0px; }
    .subtitle { font-size:18px !important; text-align: center; color: #555555; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🍅 Deteksi Daun Tomat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Pakar Deteksi Penyakit Daun Tomat Berbasis Deep Learning</p>', unsafe_allow_html=True)

# ==============================================================================
# LOAD MODEL DARI GOOGLE DRIVE
# ==============================================================================
@st.cache_resource
def load_model():
    model_path = 'BARU_model_daun_tomat_82.h5'
    
    # Cek apakah model sudah ada di lokal
    if not os.path.exists(model_path):
        with st.spinner("📥 Mengunduh model AI (26MB) dari Google Drive, mohon tunggu 1-2 menit..."):
            # File ID dari link Google Drive Anda
            file_id = "1E5tsGy0M1kQr9rgvZbuloxPWRWpKFE4i"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

# Load model
with st.spinner("🧠 Sedang menginisialisasi model AI..."):
    model = load_model()

# Kelas (sama persis dengan Colab)
class_names = ['healthy', 'early_blight', 'late_blight', 'leaf_mold']
IMG_SIZE = 128

# Label yang lebih rapi untuk tampilan
class_display_names = {
    'healthy': '🍅 HEALTHY (Sehat)',
    'early_blight': '🟤 Early Blight (Bercak Dini)',
    'late_blight': '⚫ Late Blight (Bercak Lambat)',
    'leaf_mold': '🌫️ Leaf Mold (Kapang Daun)'
}

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("📌 Informasi Sistem")
    st.markdown("""
    Aplikasi ini menggunakan **MobileNetV2 (Transfer Learning)** untuk mendeteksi 
    4 kondisi daun tomat.
    """)
    
    st.markdown("---")
    st.subheader("🌿 Kelas yang Didukung:")
    class_info = {
        'healthy': '🟢 Daun sehat tanpa bercak',
        'early_blight': '🟤 Bercak coklat kecil pada daun',
        'late_blight': '⚫ Bercak besar berwarna coklat kehitaman',
        'leaf_mold': '🌫️ Bercak abu-abu seperti jamur'
    }
    for name in class_names:
        st.markdown(f"- **{class_display_names[name]}**")
        st.caption(f"  {class_info[name]}")

# ==============================================================================
# FUNGSI PREPROCESSING (SAMA PERSIS DENGAN COLAB)
# ==============================================================================
def predict_tomato_disease(img_array, model, img_size=128):
    """
    Fungsi prediksi sama persis dengan yang di Colab
    """
    # Prapemrosesan: resize dan normalisasi
    img_resized = cv2.resize(img_array, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Tambah dimensi batch
    input_data = np.expand_dims(img_normalized, axis=0)
    
    # Prediksi
    preds = model.predict(input_data)
    pred_class_idx = np.argmax(preds[0])
    pred_label = class_names[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100
    
    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "all_probs": preds[0],
        "img_resized": img_resized
    }

# ==============================================================================
# UPLOAD GAMBAR
# ==============================================================================
uploaded_file = st.file_uploader(
    "📤 Unggah foto sampel daun tomat Anda di bawah ini:", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Buka gambar
    image = Image.open(uploaded_file)
    
    # Konversi ke BGR (OpenCV format) karena model dilatih dengan BGR
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with col1:
        st.subheader("📸 Citra Input")
        st.image(image, caption="Gambar Daun Asli", use_container_width=True)
        
    with col2:
        st.subheader("⚙️ Kontrol Analisis")
        st.write("Klik tombol di bawah ini untuk mendeteksi penyakit:")
        run_prediction = st.button("🔍 Mulai Deteksi Penyakit", type="primary", use_container_width=True)
    
    if run_prediction:
        with st.spinner("🤖 AI sedang menganalisis gambar..."):
            
            # Prediksi
            result = predict_tomato_disease(img_bgr, model, IMG_SIZE)
            pred_label = result["pred_label"]
            confidence = result["confidence"]
            all_probs = result["all_probs"]
            
            # ========== TAMPILKAN HASIL ==========
            st.write("---")
            st.subheader("📊 Hasil Diagnosa")
            
            # Metrik
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if pred_label == 'healthy':
                    st.metric(label="Status Daun", value="✅ SEHAT", delta="Normal")
                else:
                    st.metric(label="Status Daun", value="⚠️ TERINFEKSI", delta="Sakit", delta_color="inverse")
            with m_col2:
                st.metric(label="Keyakinan", value=f"{confidence:.1f}%")
            
            # Banner Status
            if pred_label == 'healthy':
                st.success(f"✅ **Hasil:** Tanaman dinyatakan **SEHAT**")
            else:
                st.error(f"⚠️ **Hasil:** Tanaman terdeteksi **{class_display_names[pred_label]}**")
            
            # ========== GRAFIK PROBABILITAS ==========
            st.write("")
            st.write("**📈 Probabilitas per Kelas:**")
            
            for idx, name in enumerate(class_names):
                prob_percentage = all_probs[idx] * 100
                display_name = class_display_names[name]
                
                st.markdown(f"🔹 {display_name}: {prob_percentage:.1f}%")
                st.progress(float(all_probs[idx]), text=f"{prob_percentage:.1f}%")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<center><small>🍅 Deteksi Penyakit Daun Tomat | CNN Transfer Learning</small></center>",
    unsafe_allow_html=True
)
