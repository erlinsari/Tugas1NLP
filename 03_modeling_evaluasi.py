import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("model", exist_ok=True)
os.makedirs("gambar", exist_ok=True)

df = pd.read_csv("data/ulasan_dana_bersih.csv")
df = df.dropna(subset=["teks_bersih", "sentimen"])
df = df[df["teks_bersih"].str.strip() != ""]

distribusi = df["sentimen"].value_counts()
print(f"Data: {len(df)} baris")
print("\nDistribusi Sentimen:")
print(distribusi.to_string())

X = df["teks_bersih"]
y = df["sentimen"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vektorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = vektorizer.fit_transform(X_train)
X_test_tfidf = vektorizer.transform(X_test)

print(f"\nData latih : {len(X_train)}")
print(f"Data uji   : {len(X_test)}")
print(f"Fitur TF-IDF: {X_train_tfidf.shape[1]}")

print("\nMelatih model SVM...")
svm = LinearSVC(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")
model = CalibratedClassifierCV(svm, cv=3)
model.fit(X_train_tfidf, y_train)
print("Model selesai dilatih.")

prediksi = model.predict(X_test_tfidf)
akurasi = accuracy_score(y_test, prediksi)
label_unik = sorted(y.unique())

print(f"\nAkurasi: {akurasi * 100:.2f}%")
print(classification_report(y_test, prediksi, target_names=label_unik))

laporan = classification_report(y_test, prediksi, target_names=label_unik, output_dict=True)
pd.DataFrame(laporan).transpose().to_csv("data/laporan_evaluasi.csv", encoding="utf-8-sig")

cm = confusion_matrix(y_test, prediksi, labels=label_unik)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
            xticklabels=label_unik, yticklabels=label_unik,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix - Sentimen Ulasan DANA", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Prediksi", fontsize=11)
ax.set_ylabel("Aktual", fontsize=11)
plt.tight_layout()
plt.savefig("gambar/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

palet = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribusi Sentimen Ulasan DANA", fontsize=14, fontweight="bold")

warna = [palet.get(k, "#6b7280") for k in distribusi.index]
batang = ax1.bar(distribusi.index, distribusi.values, color=warna, edgecolor="white", linewidth=1.5)
ax1.set_title("Jumlah Ulasan per Sentimen")
ax1.set_xlabel("Sentimen")
ax1.set_ylabel("Jumlah")
for p in batang:
    ax1.annotate(f"{p.get_height():,}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha="center", va="bottom", fontweight="bold")

ax2.pie(distribusi.values, labels=distribusi.index, colors=warna,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
ax2.set_title("Persentase Sentimen")
plt.tight_layout()
plt.savefig("gambar/distribusi_sentimen.png", dpi=150, bbox_inches="tight")
plt.close()

fig, sumbu = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Word Cloud Ulasan DANA per Sentimen", fontsize=14, fontweight="bold")

wc_cfg = {
    "Positif": {"cmap": "Greens", "ax": sumbu[0]},
    "Netral":  {"cmap": "Oranges", "ax": sumbu[1]},
    "Negatif": {"cmap": "Reds",    "ax": sumbu[2]},
}

for label, cfg in wc_cfg.items():
    teks = " ".join(df[df["sentimen"] == label]["teks_bersih"].tolist())
    if teks.strip():
        wc = WordCloud(width=500, height=300, background_color="white",
                       colormap=cfg["cmap"], max_words=80,
                       collocations=False).generate(teks)
        cfg["ax"].imshow(wc, interpolation="bilinear")
    cfg["ax"].set_title(f"Sentimen: {label}", fontsize=12, fontweight="bold")
    cfg["ax"].axis("off")

plt.tight_layout()
plt.savefig("gambar/wordcloud_sentimen.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump(model, "model/model_sentimen.pkl")
joblib.dump(vektorizer, "model/vektorizer_tfidf.pkl")

metadata = {
    "akurasi": float(akurasi),
    "jumlah_data": len(df),
    "jumlah_latih": len(X_train),
    "jumlah_uji": len(X_test),
    "jumlah_fitur": int(X_train_tfidf.shape[1]),
    "kelas": label_unik,
    "distribusi": distribusi.to_dict(),
    "algoritma": "SVM (LinearSVC + CalibratedClassifierCV)",
}
for s in label_unik:
    if s in laporan:
        metadata[f"presisi_{s.lower()}"] = laporan[s].get("precision", 0)
        metadata[f"recall_{s.lower()}"] = laporan[s].get("recall", 0)
        metadata[f"f1_{s.lower()}"] = laporan[s].get("f1-score", 0)

with open("model/metadata_model.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nModel & metadata tersimpan.")
print(f"Akurasi akhir: {akurasi * 100:.2f}%")

