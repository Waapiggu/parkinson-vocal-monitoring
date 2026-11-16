import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Parkinson Vocal Telemonitoring", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_parkinson.pkl")
    hc_baseline = np.load("models/hc_baseline.npy")
    return model, hc_baseline

model, hc_baseline = load_artifacts()

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("🎙️ Sistem Telemonitoring Stabilitas Vokal Parkinson (Berbasis MFCC)")
st.caption("Alat bantu pemantauan parameter akustik sinyal suara vokal /a/ — Bukan alat diagnosis klinis mutlak.")

with st.sidebar:
    st.header("Input Rekaman Sesi")
    session_label = st.text_input("Label Sesi / Tanggal", value=f"Sesi #{len(st.session_state['history']) + 1}")
    uploaded_file = st.file_uploader("Unggah rekaman vokal /a/ (.wav)", type=["wav"])
    
    if st.button("Reset Riwayat Monitoring"):
        st.session_state["history"] = []
        st.rerun()

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    prob_pd = model.predict_proba([mfcc_mean])[0][1] * 100
    
    if prob_pd < 40:
        status = "Stabil (Mendekati Baseline Sehat)"
        delta_color = "normal"
    elif prob_pd < 70:
        status = "Waspada (Fluktuasi Akustik Terdeteksi)"
        delta_color = "off"
    else:
        status = "Degradasi Tinggi (Perlu Evaluasi Medis)"
        delta_color = "inverse"

    if st.sidebar.button("Simpan ke Log Pemantauan", type="primary"):
        st.session_state["history"].append({
            "Sesi": session_label,
            "File": uploaded_file.name,
            "Skor Anomali (%)": round(prob_pd, 2),
            "Status": status
        })
        st.sidebar.success("Data sesi berhasil dicatat!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Indeks Anomali Vokal", f"{prob_pd:.1f}%", status, delta_color=delta_color)
    col2.metric("Durasi Efektif Fonasi", f"{len(y_trimmed)/sr:.2f} detik")
    col3.metric("Sample Rate Audio", f"{sr} Hz")

    st.divider()

    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("1. Gelombang Suara (Waveform)")
        fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y_trimmed, sr=sr, ax=ax_wave, color="#1f77b4")
        ax_wave.set_xlabel("Waktu (detik)")
        ax_wave.set_ylabel("Amplitudo")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

    with c_right:
        st.subheader("2. Profil MFCC: Pasien vs Baseline Sehat")
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(6, 3))
        koef_idx = np.arange(1, 14)
        ax_mfcc.plot(koef_idx, hc_baseline, marker="o", linestyle="--", label="Baseline Sehat (HC)", color="green")
        ax_mfcc.plot(koef_idx, mfcc_mean, marker="s", label="Rekaman Saat Ini", color="crimson")
        ax_mfcc.set_xticks(koef_idx)
        ax_mfcc.set_xlabel("Koefisien MFCC ke-n")
        ax_mfcc.set_ylabel("Nilai Energi Spektral")
        ax_mfcc.legend()
        ax_mfcc.grid(True, alpha=0.3)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)

st.divider()
st.subheader("📈 Tren Pemantauan Antar-Sesi (Telemonitoring Log)")

if len(st.session_state["history"]) > 0:
    df_hist = pd.DataFrame(st.session_state["history"])
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.line_chart(df_hist.set_index("Sesi")["Skor Anomali (%)"])
    with col_table:
        st.dataframe(df_hist, use_container_width=True)
else:
    st.info("Belum ada riwayat sesi. Unggah file .wav di panel kiri lalu klik **Simpan ke Log Pemantauan** untuk melihat grafik tren.")
