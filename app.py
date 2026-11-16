import streamlit as st
import librosa
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

st.markdown('<div class="hero-badge">BIOMEDICAL SIGNAL PROCESSING • MFCC TELEMONITORING</div>', unsafe_allow_html=True)
st.title("🎙️ NeuroVoice — Pemantauan Stabilitas Vokal Parkinson")
st.caption("Evaluasi parameter akustik fonasi vokal /a/ berbasis ekstraksi Mel-Frequency Cepstral Coefficients.")

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
        st.warning("⚠️ Durasi suara terlalu singkat (< 0.5 detik). Coba rekam ulang vokal 'AAAAA...' minimal 3 detik ya.")
    else:
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        prob_pd = model.predict_proba([mfcc_mean])[0][1] * 100
        
        if prob_pd < 40:
            status = "Stabil (Mendekati Baseline)"
            badge_color = "#00D4B4"
        elif prob_pd < 70:
            status = "Waspada (Fluktuasi Akustik)"
            badge_color = "#F59E0B"
        else:
            status = "Degradasi Tinggi (Evaluasi Medis)"
            badge_color = "#EF4444"

        if st.sidebar.button("💾 Simpan ke Log Pemantauan", type="primary", use_container_width=True):
            st.session_state["history"].append({
                "Sesi": session_label,
                "Sumber": source_name,
                "Skor Anomali (%)": round(prob_pd, 2),
                "Status": status
            })
            st.sidebar.success("Tercatat di log sesi!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Indeks Anomali Vokal", f"{prob_pd:.1f}%", status)
        c2.metric("Durasi Efektif Fonasi", f"{len(y_trimmed)/sr:.2f} detik", "Trimmed Silence")
        c3.metric("Resolusi Sampling", f"{sr:,} Hz", "13 Koefisien MFCC")

        st.markdown("<br>", unsafe_allow_html=True)

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
            fig_gauge.update_layout(
                paper_bgcolor="#162032",
                height=330,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_mfcc:
            koef_idx = [f"MFCC-{i}" for i in range(1, 14)]
            fig_mfcc = go.Figure()
            fig_mfcc.add_trace(go.Scatter(
                x=koef_idx, y=hc_baseline,
                mode='lines+markers',
                name='Baseline Sehat (HC)',
                line=dict(color='#00D4B4', width=3, dash='dash'),
                marker=dict(size=8)
            ))
            fig_mfcc.add_trace(go.Scatter(
                x=koef_idx, y=mfcc_mean,
                mode='lines+markers',
                name='Rekaman Saat Ini',
                line=dict(color='#F43F5E', width=3),
                marker=dict(size=8, symbol='square')
            ))
            fig_mfcc.update_layout(
                title="<b>Perbandingan Spektral MFCC vs Baseline Sehat</b>",
                paper_bgcolor="#162032",
                plot_bgcolor="#0B1120",
                font=dict(color="#F1F5F9"),
                height=330,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Energi Spektral", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_mfcc, use_container_width=True)

        step = max(1, len(y_trimmed) // 3000)
        y_plot = y_trimmed[::step]
        t_plot = np.linspace(0, len(y_trimmed)/sr, num=len(y_plot))

        fig_wave = go.Figure(go.Scatter(
            x=t_plot, y=y_plot,
            mode='lines',
            line=dict(color='#38BDF8', width=1.2),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.1)'
        ))
        fig_wave.update_layout(
            title="<b>Sinyal Domain Waktu (Trimmed Waveform)</b>",
            paper_bgcolor="#162032",
            plot_bgcolor="#0B1120",
            font=dict(color="#F1F5F9"),
            height=230,
            margin=dict(l=20, r=20, t=45, b=20),
            xaxis=dict(title="Waktu (detik)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Amplitudo", gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_wave, use_container_width=True)

else:
    st.info("👈 Pilih tab **Mikrofon Langsung** untuk merekam suara secara *real-time*, atau **Unggah File** untuk menguji dataset `.wav`.")

st.markdown("---")
st.subheader("📈 Tren Pemantauan Antar-Sesi (Telemonitoring Log)")

if len(st.session_state["history"]) > 0:
    df_hist = pd.DataFrame(st.session_state["history"])
    col_chart, col_table = st.columns([1.4, 1])
    
    with col_chart:
        fig_trend = go.Figure(go.Scatter(
            x=df_hist["Sesi"],
            y=df_hist["Skor Anomali (%)"],
            mode='lines+markers+text',
            text=df_hist["Skor Anomali (%)"].astype(str) + "%",
            textposition="top center",
            line=dict(color='#00D4B4', width=3),
            marker=dict(size=10, color='#F1F5F9', line=dict(width=2, color='#00D4B4'))
        ))
        fig_trend.update_layout(
            paper_bgcolor="#162032",
            plot_bgcolor="#0B1120",
            font=dict(color="#F1F5F9"),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 110], title="Skor Anomali (%)", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_table:
        st.dataframe(df_hist, use_container_width=True, height=300)
else:
    st.caption("Belum ada sesi yang disimpan. Klik **Simpan ke Log Pemantauan** setelah merekam atau mengunggah audio.")
