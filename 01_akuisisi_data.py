# Akuisisi Data Ulasan DANA

import pandas as pd
from google_play_scraper import reviews, Sort
import time
import os

# Konfigurasi
APP_ID = "id.dana"
TARGET = 2000
OUTPUT = "data/ulasan_dana_mentah.csv"

os.makedirs("data", exist_ok=True)

semua_ulasan = []
token = None

print("=" * 55)
print("  AKUISISI DATA ULASAN DANA - GOOGLE PLAY STORE")
print("=" * 55)
print(f"App ID : {APP_ID}")
print(f"Target : {TARGET} ulasan\n")

try:
    while len(semua_ulasan) < TARGET:
        sisa = TARGET - len(semua_ulasan)
        batch = min(200, sisa)

        hasil, token = reviews(
            APP_ID,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=batch,
            continuation_token=token
        )

        if not hasil:
            break

        semua_ulasan.extend(hasil)
        print(f"  Terkumpul: {len(semua_ulasan)}/{TARGET}")

        if token is None:
            break

        time.sleep(1)

except Exception as e:
    print(f"Error: {e}")

print(f"\nTotal ulasan: {len(semua_ulasan)}")

# Buat DataFrame
daftar = []
for u in semua_ulasan:
    daftar.append({
        "id_ulasan": u.get("reviewId", ""),
        "nama_pengguna": u.get("userName", ""),
        "isi_ulasan": u.get("content", ""),
        "bintang": u.get("score", 0),
        "tanggal_ulasan": u.get("at", ""),
        "jumlah_like": u.get("thumbsUpCount", 0),
    })

df = pd.DataFrame(daftar)


def tentukan_sentimen(bintang):
    if bintang >= 4:
        return "Positif"
    elif bintang == 3:
        return "Netral"
    else:
        return "Negatif"


df["sentimen"] = df["bintang"].apply(tentukan_sentimen)
df = df[df["isi_ulasan"].str.strip() != ""]
df = df.dropna(subset=["isi_ulasan"])

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Disimpan ke: {OUTPUT} ({len(df)} baris)")
print("\nDistribusi Sentimen:")
print(df["sentimen"].value_counts().to_string())
