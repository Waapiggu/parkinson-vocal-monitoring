import os
import glob
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

os.makedirs("models", exist_ok=True)

def extract_all_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # 1. Ekstraksi 13 MFCC
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # 2. Ekstraksi Jitter, Shimmer, dan Pitch F0 via Praat
    sound = parselmouth.Sound(y_trimmed, sampling_frequency=sr)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    
    acoustic_feats = np.nan_to_num([jitter_local, shimmer_local, mean_f0], nan=0.0)
    return np.concatenate([mfcc_mean, acoustic_feats])

X, y, groups, records = [], [], [], []
hc_features = []

print("Mengekstrak 13 MFCC + Jitter, Shimmer, dan Pitch F0...")

for label_idx, folder_name in enumerate(["HC", "PD"]):
    for file in glob.glob(f"dataset/{folder_name}/*.wav"):
        feats = extract_all_features(file)
        fname = os.path.basename(file)
        raw_id = os.path.splitext(fname)[0].split('_')[0]
        subject_id = f"{folder_name}_{raw_id}"
        
        X.append(feats)
        y.append(label_idx)
        groups.append(subject_id)
        if label_idx == 0:
            hc_features.append(feats)
            
        row = {
            "file": fname,
            "subject_id": subject_id,
            "label": label_idx,
            **{f"mfcc_{i+1}": feats[i] for i in range(13)},
            "jitter_percent": feats[13],
            "shimmer_percent": feats[14],
            "mean_f0_hz": feats[15]
        }
        records.append(row)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

pd.DataFrame(records).to_csv("models/extracted_mfcc.csv", index=False)
hc_baseline = np.mean(hc_features, axis=0)
np.save("models/hc_baseline.npy", hc_baseline)

n_splits = min(5, len(np.unique(groups)))
gkf = GroupKFold(n_splits=n_splits)
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="accuracy")

print(f"Akurasi Validasi Gabungan (Subject-Wise {n_splits}-Fold): {scores.mean()*100:.2f}%")
print(f"Baseline Sehat -> Jitter: {hc_baseline[13]:.2f}% | Shimmer: {hc_baseline[14]:.2f}% | F0: {hc_baseline[15]:.1f} Hz")

model.fit(X, y)
joblib.dump(model, "models/rf_parkinson.pkl")
print("Selesai! Model gabungan tersimpan di folder models/.")
