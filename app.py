import streamlit as st
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="NeuroVoice | Parkinson Telemonitoring",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #162032, #1E293B);
        border: 1px solid rgba(0, 212, 180, 0.25);
        padding: 16px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(0, 212, 180, 0.15);
        color: #00D4B4;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(0, 212, 180, 0.4);
        margin-bottom: 8px;
    }

    .protocol-box {
        background: #162032;
        border-left: 4px solid #00D4B4;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.92rem;
    }

    .analysis-card {
        background: rgba(22, 32, 50, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 18px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_parkinson.pkl")
    hc_baseline = np.load("models/hc_baseline.npy")
    return model, hc_baseline

model, hc_baseline = load_artifacts()

if "history" not in st.session_state:
    st.session_state["history"] = []

st.markdown('<div class="hero-badge">BIOMEDICAL SIGNAL PROCESSING • MFCC + JITTER & SHIMMER</div>', unsafe_allow_html=True)
st.title("🎙️ NeuroVoice — Pemantauan Stabilitas Vokal Parkinson")
st.caption("Analisis gabungan spektral MFCC serta perturbasi akustik Jitter dan Shimmer pada fonasi vokal /a/ berkelanjutan.")

with st.sidebar:
    st.markdown("### 🎛️ Panel Input Sesi")
    session_label = st.text_input("Label Sesi / Pasien", value=f"Sesi #{len(st.session_state['history']) + 1}")
    
    tab_mic, tab_file = st.tabs(["🎤 Mikrofon Langsung", "📁 Unggah File"])
    
    audio_source = None
    source_name = ""

    with tab_mic:
        st.markdown("""
        <div class="protocol-box">
            <b>📋 Protokol Rekaman Vokal:</b><br>
            1. Posisikan mic ±10–15 cm dari mulut.<br>
            2. Tarik napas dalam-dalam.<br>
            3. Klik ikon mic, lalu ucapkan vokal <b>"AAAAA..."</b> dengan nada datar selama <b>3–5 detik</b>.<br>
            4. Klik stop setelah selesai.
        </div>
        """, unsafe_allow_html=True)
        
        mic_audio = st.audio_input("Mulai Rekam Vokal /a/")
        if mic_audio is not None:
            audio_source = mic_audio
            source_name = f"Mic_{session_label.replace(' ', '_')}.wav"

    with tab_file:
        uploaded_file = st.file_uploader("Pilih rekaman dataset (.wav)", type=["wav"])
        if uploaded_file is not None:
            audio_source = uploaded_file
            source_name = uploaded_file.name

    st.markdown("---")
    if st.button("🗑️ Reset Riwayat Monitoring", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

if audio_source is not None:
    st.audio(audio_source, format="audio/wav")
    
    audio_source.seek(0)
    y, sr = librosa.load(audio_source, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    if len(y_trimmed) < sr * 0.5:
        st.warning("⚠️ Durasi suara terlalu singkat. Coba rekam ulang vokal 'AAAAA...' minimal 3 detik ya.")
    else:
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        sound = parselmouth.Sound(y_trimmed, sampling_frequency=sr)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_val = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer_val = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        
        acoustic_feats = np.nan_to_num([jitter_val, shimmer_val, mean_f0], nan=0.0)
        all_features = np.concatenate([mfcc_mean, acoustic_feats])
        
        prob_pd = model.predict_proba([all_features])[0][1] * 100
        
        if prob_pd < 40:
            status = "Stabil (Mendekati Baseline)"
            badge_color = "#00D4B4"
            summary_note = "Pola spektral dan kestabilan getaran pita suara berada dalam rentang normal mendekati kelompok kontrol sehat."
        elif prob_pd < 70:
            status = "Waspada (Fluktuasi Akustik)"
            badge_color = "#F59E0B"
            summary_note = "Terdeteksi fluktuasi mikro pada periodisitas atau energi spektral vokal. Disarankan pemantauan berkala."
        else:
            status = "Degradasi Tinggi (Evaluasi Medis)"
            badge_color = "#EF4444"
            summary_note = "Ditemukan deviasi akustik signifikan yang mengindikasikan pola disartria hipokinetik. Perlu rujukan evaluasi klinis lebih lanjut."

        if st.sidebar.button("💾 Simpan ke Log Pemantauan", type="primary", use_container_width=True):
            st.session_state["history"].append({
                "Sesi": session_label,
                "Sumber": source_name,
                "Skor Anomali (%)": round(prob_pd, 2),
                "Jitter (%)": round(acoustic_feats[0], 2),
                "Shimmer (%)": round(acoustic_feats[1], 2),
                "Status": status
            })
            st.sidebar.success("Tercatat di log sesi!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Indeks Anomali Gabungan", f"{prob_pd:.1f}%", status)
        c2.metric("Durasi Efektif Fonasi", f"{len(y_trimmed)/sr:.2f} detik", "Trimmed Silence")
        c3.metric("Resolusi Sampling", f"{sr:,} Hz", "13 MFCC + 3 Akustik")

        st.markdown(f"""
        <div class="analysis-card" style="border-left-color: {badge_color};">
            <b>🩺 Interpretasi Sistem:</b> {summary_note}
        </div>
        """, unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        diff_jitter = acoustic_feats[0] - hc_baseline[13]
        diff_shimmer = acoustic_feats[1] - hc_baseline[14]
        diff_f0 = acoustic_feats[2] - hc_baseline[15]

        a1.metric("Jitter Lokal (Instabilitas Frekuensi)", f"{acoustic_feats[0]:.2f}%", f"{diff_jitter:+.2f}% vs Baseline Sehat", delta_color="inverse")
        a2.metric("Shimmer Lokal (Instabilitas Amplitudo)", f"{acoustic_feats[1]:.2f}%", f"{diff_shimmer:+.2f}% vs Baseline Sehat", delta_color="inverse")
        a3.metric("Frekuensi Dasar Rata-rata (Pitch F0)", f"{acoustic_feats[2]:.1f} Hz", f"{diff_f0:+.1f} Hz vs Baseline Sehat", delta_color="off")

        jitter_desc = (
            f"<b>Tinggi ({acoustic_feats[0]:.2f}% > batas normal 1.04%)</b> — menandakan adanya ketidakstabilan siklus waktu getaran pita suara akibat kekakuan otot laring."
            if acoustic_feats[0] > 1.04 else
            f"<b>Terkendali ({acoustic_feats[0]:.2f}% ≤ batas normal 1.04%)</b> — ketegangan pita suara menjaga periode frekuensi tetap konsisten."
        )
        shimmer_desc = (
            f"<b>Meningkat ({acoustic_feats[1]:.2f}% > batas normal 3.81%)</b> — mengindikasikan fluktuasi amplitudo puncak akibat penutupan celah glotis yang kurang rapat."
            if acoustic_feats[1] > 3.81 else
            f"<b>Stabil ({acoustic_feats[1]:.2f}% ≤ batas normal 3.81%)</b> — kontrol tekanan aliran udara subglotis dan penutupan pita suara terjaga baik."
        )
        f0_desc = (
            "cenderung merendah/monoton dibanding rata-rata kelompok sehat (karakteristik <i>hypophonia</i>)."
            if diff_f0 < -15 else
            "berada dalam rentang variasi nada dasar yang wajar."
        )

        st.markdown(f"""
        <div class="analysis-card">
            <b>🔬 Analisis Perturbasi Biosinyal Glotal:</b><br>
            • <b>Jitter:</b> {jitter_desc}<br>
            • <b>Shimmer:</b> {shimmer_desc}<br>
            • <b>Pitch F0:</b> Frekuensi dasar berada di <b>{acoustic_feats[2]:.1f} Hz</b>, {f0_desc}
        </div>
        """, unsafe_allow_html=True)
        col_gauge, col_mfcc = st.columns([1, 1.6])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pd,
                number={'suffix': "%", 'font': {'size': 42, 'color': '#F1F5F9'}},
                title={'text': "<b>Skor Deviasi Akustik</b>", 'font': {'size': 16, 'color': '#94A3B8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': badge_color},
                    'bgcolor': "#0B1120",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(0, 212, 180, 0.15)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="#162032", height=330, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        with col_mfcc:
            koef_idx = [f"MFCC-{i}" for i in range(1, 14)]
            fig_mfcc = go.Figure()
            fig_mfcc.add_trace(go.Scatter(x=koef_idx, y=hc_baseline[:13], mode='lines+markers', name='Baseline Sehat (HC)', line=dict(color='#00D4B4', width=3, dash='dash'), marker=dict(size=8)))
            fig_mfcc.add_trace(go.Scatter(x=koef_idx, y=mfcc_mean, mode='lines+markers', name='Rekaman Saat Ini', line=dict(color='#F43F5E', width=3), marker=dict(size=8, symbol='square')))
            fig_mfcc.update_layout(
                title="<b>Perbandingan Spektral MFCC vs Baseline Sehat</b>",
                paper_bgcolor="#162032", plot_bgcolor="#0B1120", font=dict(color="#F1F5F9"),
                height=330, margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Energi Spektral", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_mfcc, width="stretch")

        mfcc_diffs = np.abs(mfcc_mean - hc_baseline[:13])
        max_idx = int(np.argmax(mfcc_diffs))
        max_coef_name = f"MFCC-{max_idx + 1}"
        euclid_dist = float(np.linalg.norm(mfcc_mean - hc_baseline[:13]))

        st.markdown(f"""
        <div class="analysis-card">
            <b>📊 Analisis Selubung Spektral (MFCC):</b><br>
            Jarak deviasi spektral total terhadap acuan sehat adalah <b>{euclid_dist:.2f}</b>, dengan penyimpangan terbesar terdeteksi pada koefisien <b>{max_coef_name}</b> 
            (selisih <b>{mfcc_diffs[max_idx]:.2f}</b>). Pergeseran pada koefisien orde rendah mencerminkan perubahan energi keseluruhan fonasi, sedangkan orde menengah–tinggi menunjukkan perubahan resonansi traktus vokal akibat keterbatasan gerak artikulator.
        </div>
        """, unsafe_allow_html=True)

        step = max(1, len(y_trimmed) // 3000)
        y_plot = y_trimmed[::step]
        t_plot = np.linspace(0, len(y_trimmed)/sr, num=len(y_plot))

        fig_wave = go.Figure(go.Scatter(x=t_plot, y=y_plot, mode='lines', line=dict(color='#38BDF8', width=1.2), fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)'))
        fig_wave.update_layout(
            title="<b>Sinyal Domain Waktu (Trimmed Waveform)</b>",
            paper_bgcolor="#162032", plot_bgcolor="#0B1120", font=dict(color="#F1F5F9"),
            height=230, margin=dict(l=20, r=20, t=45, b=20),
            xaxis=dict(title="Waktu (detik)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Amplitudo", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_wave, width="stretch")

        peak_amp = float(np.max(np.abs(y_trimmed)))
        rms_energy = float(np.sqrt(np.mean(y_trimmed**2)))
        st.markdown(f"""
        <div class="analysis-card">
            <b>〰️ Analisis Domain Waktu (Waveform):</b><br>
            Sinyal aktif berdurasi <b>{len(y_trimmed)/sr:.2f} detik</b> memiliki amplitudo puncak <b>{peak_amp:.3f}</b> dan energi rata-rata (RMS) <b>{rms_energy:.4f}</b>. 
            Ketidakrataan amplop gelombang pada rekaman vokal datar menandakan adanya modulasi amplitudo tak terkendali selama fase fonasi berkelanjutan.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 Pilih tab **Mikrofon Langsung** untuk merekam fonasi vokal /a/, atau **Unggah File** untuk menguji dataset `.wav`.")

st.markdown("---")
st.subheader("📈 Tren Pemantauan Antar-Sesi (Telemonitoring Log)")

if len(st.session_state["history"]) > 0:
    df_hist = pd.DataFrame(st.session_state["history"])
    col_chart, col_table = st.columns([1.3, 1.2])
    
    with col_chart:
        fig_trend = go.Figure(go.Scatter(
            x=df_hist["Sesi"], y=df_hist["Skor Anomali (%)"],
            mode='lines+markers+text', text=df_hist["Skor Anomali (%)"].astype(str) + "%", textposition="top center",
            line=dict(color='#00D4B4', width=3), marker=dict(size=10, color='#F1F5F9', line=dict(width=2, color='#00D4B4'))
        ))
        fig_trend.update_layout(
            paper_bgcolor="#162032", plot_bgcolor="#0B1120", font=dict(color="#F1F5F9"),
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 110], title="Skor Anomali (%)", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_trend, width="stretch")
        
    with col_table:
        st.dataframe(df_hist, width="stretch", height=300)
else:
    st.caption("Belum ada sesi yang disimpan. Klik **Simpan ke Log Pemantauan** setelah merekam atau mengunggah audio.")
