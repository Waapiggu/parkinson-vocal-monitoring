import os
import glob
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

os.makedirs("models", exist_ok=True)

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

X, y, groups, records = [], [], [], []
hc_features = []

print("Mengekstrak fitur MFCC dari dataset...")

# 1. Proses data Healthy Control (Label 0)
for file in glob.glob("dataset/HC/*.wav"):
    feat = extract_mfcc(file)
    fname = os.path.basename(file)
    raw_id = os.path.splitext(fname)[0].split('_')[0]
    subject_id = f"HC_{raw_id}"
    
    X.append(feat)
    y.append(0)
    groups.append(subject_id)
    hc_features.append(feat)
    records.append({"file": fname, "subject_id": subject_id, "label": 0, **{f"mfcc_{i+1}": feat[i] for i in range(13)}})

# 2. Proses data Parkinson Disease (Label 1)
for file in glob.glob("dataset/PD/*.wav"):
    feat = extract_mfcc(file)
    fname = os.path.basename(file)
    raw_id = os.path.splitext(fname)[0].split('_')[0]
    subject_id = f"PD_{raw_id}"
    
    X.append(feat)
    y.append(1)
    groups.append(subject_id)
    records.append({"file": fname, "subject_id": subject_id, "label": 1, **{f"mfcc_{i+1}": feat[i] for i in range(13)}})

if len(X) == 0:
    raise RuntimeError("File .wav tidak ditemukan di dataset/HC/ atau dataset/PD/.")

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# Simpan tabel fitur ringan (CSV) buat Anggota 2
df_features = pd.DataFrame(records)
df_features.to_csv("models/extracted_mfcc.csv", index=False)

# Simpan rata-rata profil MFCC orang sehat (HC Baseline)
hc_baseline = np.mean(hc_features, axis=0)
np.save("models/hc_baseline.npy", hc_baseline)

# Validasi bebas kebocoran data (Subject-Wise GroupKFold)
n_splits = min(5, len(np.unique(groups)))
gkf = GroupKFold(n_splits=n_splits)
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="accuracy")
print(f"Total Sampel: {len(X)} | Subjek Unik: {len(np.unique(groups))}")
print(f"Akurasi Validasi (Subject-Wise {n_splits}-Fold): {scores.mean()*100:.2f}%")

# Latih model final & simpan
model.fit(X, y)
joblib.dump(model, "models/rf_parkinson.pkl")
print("Selesai! Model, Baseline, dan CSV fitur tersimpan di folder models/.")
