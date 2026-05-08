import streamlit as st
from PIL import Image
import numpy as np
import time

# ===== CONFIG =====
st.set_page_config(page_title="Deteksi Daun Tomat", layout="wide")

# ===== TITLE =====
st.title("🌿 Deteksi Penyakit Daun Tomat")
st.write("Upload gambar daun untuk mendeteksi penyakit dan melihat tahapan pemrosesan citra")

# ===== SIDEBAR =====
st.sidebar.header("📤 Upload Gambar")
file = st.sidebar.file_uploader(
    "Pilih gambar",
    type=["jpg", "png", "jpeg", "webp"]
)

# ===== MAIN =====
if file is not None:
    image = Image.open(file)

    # ===== LOADING =====
    with st.sidebar:
        st.write("⏳ Gambar berhasil diupload")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)
        st.success("Siap diproses!")

    # ===== LAYOUT =====
    col_img, col_info = st.columns([2, 1])

    # =========================
    # ===== KIRI: GAMBAR =====
    # =========================
    with col_img:
        st.subheader("📷 Visualisasi Citra")

        tab1, tab2 = st.tabs(["Original", "Processed (Pipeline)"])

        # ===== ORIGINAL =====
        with tab1:
            st.image(image, width="stretch")

        # ===== PROCESSED PIPELINE =====
        with tab2:
            st.write("🔬 Tahapan Pemrosesan Citra")

            # ===== STEP 1: RESIZE =====
            img = image.resize((224, 224))
            st.subheader("1️⃣ Resize")
            st.caption("Mengubah ukuran gambar menjadi 224x224 agar sesuai input model")
            st.image(img, width="stretch")

            # ===== STEP 2: GRAYSCALE =====
            img_np = np.array(img)
            gray = np.mean(img_np, axis=2)
            st.subheader("2️⃣ Grayscale")
            st.caption("Mengubah citra menjadi skala abu-abu untuk memudahkan analisis")
            st.image(gray, width="stretch", clamp=True)

            # ===== STEP 3: ENHANCEMENT =====
            enhanced = np.clip(gray * 1.5, 0, 255)
            st.subheader("3️⃣ Enhancement (Contrast)")
            st.caption("Meningkatkan kontras citra agar fitur lebih terlihat")
            st.image(enhanced, width="stretch", clamp=True)

            # ===== STEP 4: SEGMENTATION =====
            threshold = enhanced > 120
            segmented = threshold * 255
            st.subheader("4️⃣ Segmentation")
            st.caption("Memisahkan objek daun dari background menggunakan threshold")
            st.image(segmented, width="stretch", clamp=True)

            # ===== STEP 5: MORPHOLOGY =====
            morph = segmented.copy()
            morph[morph < 255] = 0
            st.subheader("5️⃣ Morphology")
            st.caption("Menghilangkan noise untuk memperjelas objek")
            st.image(morph, width="stretch", clamp=True)

    # =========================
    # ===== KANAN: INFO =====
    # =========================
    with col_info:
        st.subheader("📊 Hasil Deteksi")

        if st.button("🔍 Deteksi Penyakit"):
            with st.spinner("Menganalisis gambar..."):
                time.sleep(1)

                # ===== HASIL DUMMY =====
                hasil = "Early Blight"
                conf = 0.92

            # ===== OUTPUT =====
            if "Healthy" in hasil:
                st.success(f"✅ {hasil}")
            else:
                st.error(f"⚠️ {hasil}")

            st.metric("Confidence", f"{conf*100:.2f}%")

            # ===== INFORMASI =====
            st.markdown("### 📌 Informasi Penyakit")
            st.write("""
            **Early Blight** adalah penyakit yang disebabkan oleh jamur *Alternaria solani*.

            **Gejala:**
            - Bercak coklat pada daun
            - Daun menguning
            - Daun mengering

            **Penanganan:**
            - Gunakan fungisida
            - Pangkas daun yang terinfeksi
            - Jaga kelembaban tanaman
            """)

else:
    st.info("Silakan upload gambar di sidebar 👈")

# ===== FOOTER =====
st.markdown("---")
st.caption("Project Deteksi Penyakit Daun Tomat 🌿 | Kelompok Kamu")