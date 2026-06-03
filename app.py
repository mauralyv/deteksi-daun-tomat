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
    .stButton > button { background-color: #E74C3C; color: white; border-radius: 10px; padding: 10px 24px; }
    .stButton > button:hover { background-color: #c0392b; color: white; }
    /* Sembunyikan ikon plus default */
    .stFileUploader > div > button > svg { display: none; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🍅 Deteksi Penyakit Daun Tomat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Pakar Deteksi Penyakit Daun Tomat Berbasis Deep Learning</p>', unsafe_allow_html=True)

# ==============================================================================
# LOAD MODEL DARI GOOGLE DRIVE
# ==============================================================================
@st.cache_resource
def load_model():
    model_path = 'BARU_model_daun_tomat_82.h5'
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Mengunduh model AI (25MB), mohon tunggu..."):
            file_id = "1E5tsGy0M1kQr9rgvZbuloxPWRWpKFE4i"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

with st.spinner("🧠 Menginisialisasi model AI..."):
    model = load_model()

# ==============================================================================
# KELAS YANG DIDETEKSI
# ==============================================================================
class_names = ['healthy', 'yellow_leaf', 'late_blight', 'leaf_mold']

class_display_names = {
    'healthy': '🍅 HEALTHY (Sehat)',
    'yellow_leaf': '🟡 Yellow Leaf Curl Virus (Daun Menguning)',
    'late_blight': '⚫ Late Blight (Bercak Coklat)',
    'leaf_mold': '🌫️ Leaf Mold (Jamur Daun)'
}

class_info = {
    'healthy': '🟢 Daun sehat tanpa bercak',
    'yellow_leaf': '🟡 Daun menguning dan menggulung',
    'late_blight': '⚫ Bercak besar berwarna coklat kehitaman',
    'leaf_mold': '🌫️ Bercak abu-abu seperti jamur'
}

IMG_SIZE = 128

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
    for name in class_names:
        st.markdown(f"- **{class_display_names[name]}**")
        st.caption(f"  {class_info[name]}")
    
    st.markdown("---")
    st.caption("💡 **Tips:** Pastikan gambar daun terlihat jelas dan pencahayaan cukup.")

# ==============================================================================
# FUNGSI PREPROCESSING & PREDIKSI
# ==============================================================================
def predict_tomato_disease(img_array, model, img_size=128):
    img_resized = cv2.resize(img_array, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(img_normalized, axis=0)
    
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
# SESSION STATE UNTUK MENYIMPAN GAMBAR
# ==============================================================================
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None

# ==============================================================================
# UPLOAD GAMBAR DENGAN 2 OPSI (Tombol + Drag & Drop)
# ==============================================================================
st.subheader("📤 Pilih Gambar Daun Tomat")

col_up1, col_up2 = st.columns([1, 3])

with col_up1:
    # Tombol untuk upload (lebih jelas dari ikon plus)
    uploaded_file = st.file_uploader(
        "Pilih gambar dari komputer:", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_up2:
    st.markdown("**ATAU**")
    st.markdown("📂 **Drag & drop** gambar ke kotak di bawah ini")

# Jika ada file yang diupload
if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.session_state.image_bytes = uploaded_file.getvalue()

# Tampilkan gambar jika ada
if st.session_state.uploaded_image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Citra Input")
        st.image(st.session_state.uploaded_image, caption="Gambar Daun Asli", use_container_width=True)
        
        # Tombol untuk ganti gambar (jelas fungsinya)
        if st.button("🔄 Ganti Gambar", use_container_width=True):
            st.session_state.uploaded_image = None
            st.session_state.image_bytes = None
            st.rerun()
    
    with col2:
        st.subheader("⚙️ Kontrol Analisis")
        run_prediction = st.button("🔍 Mulai Deteksi Penyakit", type="primary", use_container_width=True)
    
    if run_prediction:
        with st.spinner("🤖 AI sedang menganalisis gambar..."):
            # Konversi ke array
            img_array = np.array(st.session_state.uploaded_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            result = predict_tomato_disease(img_bgr, model, IMG_SIZE)
            pred_label = result["pred_label"]
            confidence = result["confidence"]
            all_probs = result["all_probs"]
            
            st.markdown("---")
            st.subheader("📊 Hasil Diagnosa")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if pred_label == 'healthy':
                    st.metric(label="Status Daun", value="✅ SEHAT", delta="Normal")
                else:
                    st.metric(label="Status Daun", value="⚠️ TERINFEKSI", delta="Sakit", delta_color="inverse")
            with col_m2:
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.1f}%")
            
            if pred_label == 'healthy':
                st.success(f"✅ **Hasil:** Tanaman dinyatakan **SEHAT**. Tidak diperlukan tindakan khusus.")
            else:
                st.error(f"⚠️ **Hasil:** Tanaman terdeteksi **{class_display_names[pred_label]}**")
                st.info(f"💡 **Rekomendasi:** {class_info[pred_label]}. Segera lakukan tindakan pengendalian.")
            
            # ========== GRAFIK PROBABILITAS ==========
            st.write("")
            st.write("**📈 Probabilitas per Kelas:**")
            
            for idx, name in enumerate(class_names):
                prob = all_probs[idx] * 100
                display_name = class_display_names[name]
                
                st.markdown(f"🔹 {display_name}: {prob:.1f}%")
                st.progress(int(prob), text=f"{prob:.1f}%")
else:
    # Tampilkan placeholder jika belum ada gambar
    st.info("👈 Silakan pilih gambar daun tomat terlebih dahulu")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<center><small>🍅 Deteksi Penyakit Daun Tomat | MobileNetV2 Transfer Learning</small></center>",
    unsafe_allow_html=True
)
