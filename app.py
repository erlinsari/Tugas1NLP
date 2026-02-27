import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os, re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="SentiDana", page_icon="💳", layout="wide", initial_sidebar_state="collapsed")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def muat_model():
    return joblib.load("model/model_sentimen.pkl"), joblib.load("model/vektorizer_tfidf.pkl")

@st.cache_data
def muat_data():
    return pd.read_csv("data/ulasan_dana_bersih.csv")

@st.cache_data
def muat_meta():
    with open("model/metadata_model.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def muat_nlp():
    return StopWordRemoverFactory().create_stop_word_remover(), StemmerFactory().create_stemmer()

SLANG = {
    "gk":"tidak","ga":"tidak","gak":"tidak","ngga":"tidak","nggak":"tidak",
    "yg":"yang","dgn":"dengan","sy":"saya","sdh":"sudah","udh":"sudah",
    "udah":"sudah","blm":"belum","gmn":"bagaimana","gimana":"bagaimana",
    "krn":"karena","karna":"karena","tp":"tapi","tpi":"tapi",
    "bgt":"sangat","banget":"sangat","aja":"saja","jg":"juga",
    "lg":"lagi","klo":"kalau","kalo":"kalau","bs":"bisa","dr":"dari",
    "utk":"untuk","tdk":"tidak","km":"kamu","lu":"kamu",
    "gue":"saya","gw":"saya","abis":"habis","lbh":"lebih",
    "emang":"memang","org":"orang","tau":"tahu",
    "aplikasinya":"aplikasi","appnya":"aplikasi",
}

WARNA = {"Positif": "#16a34a", "Negatif": "#dc2626", "Netral": "#d97706"}

def preprocess(teks, sw, st_):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", "", teks)
    teks = re.sub(r"@\w+|#\w+", "", teks)
    teks = teks.encode("ascii", "ignore").decode("ascii")
    teks = re.sub(r"\d+", "", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    teks = " ".join([SLANG.get(k, k) for k in teks.split()])
    teks = sw.remove(teks)
    teks = st_.stem(teks)
    return teks

if not (os.path.exists("model/model_sentimen.pkl") and os.path.exists("data/ulasan_dana_bersih.csv")):
    st.error("Model/data belum tersedia.")
    st.stop()

model, vektorizer = muat_model()
data = muat_data()
meta = muat_meta()
sw, st_ = muat_nlp()

st.markdown("""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-logo">DANA</div>
        <div class="nav-title">SentiDana</div>
    </div>
</div>
""", unsafe_allow_html=True)

halaman = st.radio("nav", ["Dashboard", "Analisis Sentimen", "Grafik dan Statistik", "Data Ulasan"],
                    horizontal=True, label_visibility="collapsed")

if halaman == "Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="h-title">Analisis Sentimen<br>Ulasan DANA</div>
        <div class="h-desc">
            NLP Pipeline lengkap untuk menganalisis sentimen ulasan
            aplikasi DANA dari Google Play Store.
        </div>
    </div>
    """, unsafe_allow_html=True)

    akurasi = meta.get("akurasi", 0)
    total = meta.get("jumlah_data", 0)
    fitur = meta.get("jumlah_fitur", 0)
    dist = meta.get("distribusi", {})
    pos_pct = round(dist.get("Positif", 0) / total * 100, 1) if total else 0

    for col, val, lbl, blue in zip(
        st.columns(4),
        [f"{akurasi*100:.1f}%", f"{total:,}", f"{fitur:,}", f"{pos_pct}%"],
        ["Akurasi Model", "Total Ulasan", "Fitur TF-IDF", "Ulasan Positif"],
        [True, False, False, False]
    ):
        with col:
            cls = " blue" if blue else ""
            st.markdown(f'<div class="stat-card"><div class="sc-label">{lbl}</div><div class="sc-value{cls}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Pipeline NLP</div>', unsafe_allow_html=True)
    pipe = [
        ("01", "Data Acquisition", "Scraping 2.000 ulasan DANA dari Google Play Store"),
        ("02", "Text Cleaning", "Hapus URL, mention, emoji, angka, karakter khusus"),
        ("03", "Pre-processing", "Normalisasi slang, stopword removal, stemming"),
        ("04", "Feature Engineering", "TF-IDF Vectorizer (unigram + bigram)"),
        ("05", "Modeling", "SVM (LinearSVC) dengan class weight balanced"),
        ("06", "Evaluation", "Accuracy, precision, recall, F1-score"),
    ]
    ca, cb = st.columns(2)
    for i, (n, t, d) in enumerate(pipe):
        with (ca if i < 3 else cb):
            st.markdown(f'<div class="timeline-item"><div class="tl-dot">{n}</div><div><div class="tl-title">{t}</div><div class="tl-desc">{d}</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 3])
    with col_l:
        for lbl, jml in dist.items():
            pct = jml / total * 100 if total else 0
            w = WARNA.get(lbl, "#64748b")
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px"><span style="color:#64748b">{lbl}</span><span style="color:{w};font-weight:700">{jml:,}</span></div>', unsafe_allow_html=True)
            st.progress(min(pct / 100, 1.0))
    with col_r:
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="none")
        ax.pie(list(dist.values()), labels=list(dist.keys()),
               colors=[WARNA.get(l, "#64748b") for l in dist.keys()],
               autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2.5),
               textprops=dict(color="#475569", fontsize=10))
        fig.patch.set_alpha(0)
        st.pyplot(fig, transparent=True)
        plt.close()

elif halaman == "Analisis Sentimen":
    st.markdown('<div class="page-label">Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analisis Sentimen Ulasan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Masukkan teks ulasan DANA untuk menganalisis sentimennya secara otomatis.</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        ulasan = st.text_area("Teks ulasan", placeholder="Contoh: Aplikasi DANA sangat membantu transaksi digital...", height=160)
        btn = st.button("Analisis Sentimen", type="primary")

    with col_out:
        if btn and ulasan.strip():
            bersih = preprocess(ulasan, sw, st_)
            hasil = model.predict(vektorizer.transform([bersih]))[0]
            proba = model.predict_proba(vektorizer.transform([bersih]))[0]
            kelas = model.classes_

            css = {"Positif": "rc-positive", "Negatif": "rc-negative", "Netral": "rc-neutral"}
            clr = {"Positif": "rc-pos-color", "Negatif": "rc-neg-color", "Netral": "rc-net-color"}
            st.markdown(f'<div class="result-card {css.get(hasil)}"><div class="rc-label">Hasil Prediksi</div><div class="rc-value {clr.get(hasil)}">{hasil}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem'>Tingkat Kepercayaan</div>", unsafe_allow_html=True)
            for i, k in enumerate(kelas):
                p = proba[i]
                w = WARNA.get(k, "#64748b")
                st.markdown(f'<div class="conf-bar"><span class="conf-label">{k}</span><div class="conf-track"><div class="conf-fill" style="width:{p*100:.1f}%;background:{w}"></div></div><span class="conf-pct">{p*100:.1f}%</span></div>', unsafe_allow_html=True)

            with st.expander("Detail Pre-processing"):
                st.code(ulasan, language=None)
                st.code(bersih, language=None)
        elif btn:
            st.warning("Isi teks ulasan terlebih dahulu.")
        else:
            st.markdown('<div class="placeholder-box"><div class="ph-title">Belum ada teks yang dianalisis</div><div class="ph-desc">Masukkan ulasan lalu klik Analisis Sentimen.</div></div>', unsafe_allow_html=True)

elif halaman == "Grafik dan Statistik":
    st.markdown('<div class="page-label">Visualisasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Grafik dan Statistik</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Ringkasan visual dari hasil analisis sentimen ulasan DANA.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Performa Model</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{meta.get('akurasi',0)*100:.1f}%")
    for col, s in zip([m2, m3, m4], ["Positif", "Negatif", "Netral"]):
        f1 = meta.get(f"f1_{s.lower()}", 0)
        col.metric(f"F1 {s}", f"{f1*100:.1f}%")

    st.markdown('<div class="section-title">Distribusi Rating dan Confusion Matrix</div>', unsafe_allow_html=True)
    col_rating, col_cm = st.columns(2)

    with col_rating:
        if "bintang" in data.columns:
            rating_count = data["bintang"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            warna_rating = ["#dc2626", "#f97316", "#f59e0b", "#84cc16", "#16a34a"]
            bars = ax.bar(rating_count.index, rating_count.values, color=warna_rating, edgecolor="white", linewidth=1.2)
            ax.set_xlabel("Rating Bintang", fontsize=10)
            ax.set_ylabel("Jumlah Ulasan", fontsize=10)
            ax.set_title("Distribusi Rating Ulasan", fontsize=12, fontweight="bold")
            ax.set_xticks([1, 2, 3, 4, 5])
            for b in bars:
                ax.annotate(f"{b.get_height():,}", (b.get_x() + b.get_width()/2., b.get_height()),
                            ha="center", va="bottom", fontweight="bold", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_cm:
        if os.path.exists("gambar/confusion_matrix.png"):
            st.image("gambar/confusion_matrix.png", use_container_width=True)

    st.markdown('<div class="section-title">Analisis Kata</div>', unsafe_allow_html=True)
    col_wc, col_kata = st.columns([3, 2])

    with col_wc:
        if os.path.exists("gambar/wordcloud_sentimen.png"):
            st.image("gambar/wordcloud_sentimen.png", use_container_width=True)

    with col_kata:
        pilih_sent = st.selectbox("Pilih sentimen", ["Positif", "Negatif", "Netral"], key="kata_sent")
        df_sent = data[data["sentimen"] == pilih_sent]
        if "teks_bersih" in df_sent.columns and len(df_sent) > 0:
            semua_kata = " ".join(df_sent["teks_bersih"].dropna().tolist()).split()
            from collections import Counter
            freq = Counter(semua_kata).most_common(15)
            if freq:
                kata_df = pd.DataFrame(freq, columns=["Kata", "Frekuensi"])
                kata_df.index = range(1, len(kata_df) + 1)
                st.dataframe(kata_df, use_container_width=True, height=350)

    st.markdown('<div class="section-title">Detail Metrik per Sentimen</div>', unsafe_allow_html=True)
    for s in ["Positif", "Negatif", "Netral"]:
        p = meta.get(f"presisi_{s.lower()}", 0)
        r = meta.get(f"recall_{s.lower()}", 0)
        f1 = meta.get(f"f1_{s.lower()}", 0)
        w = WARNA.get(s, "#64748b")
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-left:4px solid {w};
                    border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:600;color:#1a202c;font-size:0.9rem">{s}</span>
            <div style="display:flex;gap:24px;font-size:0.82rem;color:#64748b">
                <span>Presisi <strong style="color:#1a202c">{p*100:.1f}%</strong></span>
                <span>Recall <strong style="color:#1a202c">{r*100:.1f}%</strong></span>
                <span>F1 <strong style="color:{w}">{f1*100:.1f}%</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif halaman == "Data Ulasan":
    st.markdown('<div class="page-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Data Ulasan DANA</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Jelajahi dataset ulasan yang digunakan untuk melatih model.</div>', unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns(3)
    with cf1: f_sent = st.selectbox("Sentimen", ["Semua", "Positif", "Netral", "Negatif"])
    with cf2: f_star = st.selectbox("Bintang", ["Semua", 1, 2, 3, 4, 5])
    with cf3: n_rows = st.slider("Jumlah baris", 10, 300, 50)

    df_f = data.copy()
    if f_sent != "Semua": df_f = df_f[df_f["sentimen"] == f_sent]
    if f_star != "Semua": df_f = df_f[df_f["bintang"] == f_star]

    tampil = df_f[["isi_ulasan", "bintang", "sentimen", "tanggal_ulasan"]].head(n_rows)
    tampil.columns = ["Ulasan", "Bintang", "Sentimen", "Tanggal"]
    st.dataframe(tampil, use_container_width=True, height=420)

    csv = df_f.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("Download CSV", csv, file_name="ulasan_dana.csv", mime="text/csv")
