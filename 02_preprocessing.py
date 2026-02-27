# Text Cleaning & Preprocessing

import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

INPUT = "data/ulasan_dana_mentah.csv"
OUTPUT = "data/ulasan_dana_bersih.csv"

df = pd.read_csv(INPUT)
print(f"Data dimuat: {len(df)} baris\n")

print("Memuat Stemmer & StopWord...")
hapus_sw = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

kamus_slang = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak",
    "ngga": "tidak", "nggak": "tidak", "kagak": "tidak",
    "yg": "yang", "yng": "yang", "drpd": "daripada",
    "dgn": "dengan", "dg": "dengan", "sm": "sama",
    "sy": "saya", "gue": "saya", "gw": "saya",
    "lo": "kamu", "lu": "kamu", "km": "kamu",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "blom": "belum", "blum": "belum",
    "dpt": "dapat", "bsa": "bisa", "bs": "bisa",
    "gmn": "bagaimana", "gimana": "bagaimana",
    "hrs": "harus", "krn": "karena", "karna": "karena",
    "tp": "tapi", "tpi": "tapi", "ttg": "tentang",
    "dl": "dulu", "dlu": "dulu",
    "bnyk": "banyak", "byk": "banyak",
    "msh": "masih", "masi": "masih",
    "bgt": "sangat", "banget": "sangat",
    "aja": "saja", "aj": "saja",
    "jg": "juga", "spt": "seperti",
    "lg": "lagi", "klo": "kalau", "kalo": "kalau",
    "dr": "dari", "utk": "untuk", "u": "untuk",
    "tdk": "tidak", "pd": "pada",
    "abis": "habis", "lbh": "lebih", "lbih": "lebih",
    "org": "orang", "emg": "memang", "emang": "memang",
    "knp": "kenapa", "tau": "tahu",
    "jgn": "jangan", "aplikasinya": "aplikasi", "appnya": "aplikasi",
}


def normalisasi_slang(teks):
    return " ".join([kamus_slang.get(k, k) for k in teks.split()])


def bersihkan_teks(teks):
    if pd.isna(teks) or str(teks).strip() == "":
        return ""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", "", teks)
    teks = re.sub(r"@\w+|#\w+", "", teks)
    teks = teks.encode("ascii", "ignore").decode("ascii")
    teks = re.sub(r"\d+", "", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    teks = normalisasi_slang(teks)
    teks = hapus_sw.remove(teks)
    teks = stemmer.stem(teks)
    return teks.strip()


print("Memproses teks...")
total = len(df)
hasil = []

for i, baris in df.iterrows():
    hasil.append(bersihkan_teks(baris["isi_ulasan"]))
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{total} selesai...")

df["teks_bersih"] = hasil
df = df[df["teks_bersih"].str.strip() != ""]
df = df.dropna(subset=["teks_bersih"])

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nSelesai. {len(df)} baris -> {OUTPUT}")
print("\nDistribusi Sentimen:")
print(df["sentimen"].value_counts().to_string())
