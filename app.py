"""
Dashboard Prediksi Tingkat Obesitas
Menggunakan model XGBoost (dan ANN opsional) yang sudah dilatih
pada notebook obesity_model_v2.ipynb

File model yang dibutuhkan (letakkan di folder yang sama dengan app.py):
- xgb_model.pkl
- scaler.pkl
- encoders.pkl
- label_encoder_target.pkl
- feature_cols.pkl
- ann_model.keras   (opsional, jika ingin menampilkan prediksi ANN juga)
- model_metrics.csv (opsional, untuk halaman perbandingan model)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="⚕️",
    layout="wide",
)

# ------------------------------------------------------------------
# CUSTOM CSS — TEMA PALET #06142E #1B3358 #473E66 #BD83B8 #F5D7DB #F1916D
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Palet warna ──────────────────────────────────────────
       #06142E  navy gelap
       #1B3358  biru navy
       #473E66  ungu tua
       #BD83B8  ungu muda / lavender
       #F5D7DB  pink blush (background utama)
       #F1916D  oranye salmon (aksen)
    ── ─────────────────────────────────────────────────────── */

    /* Background utama */
    .stApp {
        background: linear-gradient(160deg, #F5D7DB 0%, #e8c8d4 40%, #d9b8cc 100%);
    }

    /* Teks utama */
    html, body, [class*="css"] {
        color: #06142E;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B3358 0%, #06142E 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #F5D7DB !important;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(120deg, #473E66 0%, #1B3358 60%, #06142E 100%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(6, 20, 46, 0.30);
    }
    .hero-banner h1 {
        color: #F5D7DB !important;
        margin: 0;
        font-size: 2rem;
    }
    .hero-banner p {
        color: #BD83B8;
        margin: 6px 0 0 0;
        font-size: 1rem;
    }

    /* Card umum */
    .pastel-card {
        background: rgba(255,255,255,0.75);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 14px rgba(6, 20, 46, 0.10);
        border: 1px solid #BD83B8;
        margin-bottom: 16px;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.80);
        border-radius: 18px;
        padding: 24px 28px;
        border: 1px solid #BD83B8;
        box-shadow: 0 6px 18px rgba(6, 20, 46, 0.10);
    }

    /* Buttons */
    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(120deg, #F1916D, #e07a56) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(241, 145, 109, 0.40);
        transition: transform 0.15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(241, 145, 109, 0.55);
    }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F5D7DB, #ecc8ce);
        border-radius: 14px;
        padding: 12px 16px;
        border: 1px solid #BD83B8;
        text-align: center;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #06142E !important;
        text-align: center;
    }

    /* Headings */
    h1, h2, h3 {
        color: #1B3358 !important;
    }

    /* Selectbox / input borders */
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stSlider {
        border-radius: 10px !important;
    }

    /* Sidebar caption */
    .sidebar-caption {
        background: rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #F5D7DB !important;
    }

    /* Dataframe header */
    [data-testid="stDataFrame"] th {
        background-color: #473E66 !important;
        color: #F5D7DB !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #473E66 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #F1916D !important;
        color: #06142E !important;
    }

    /* Expander */
    details summary {
        color: #1B3358 !important;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border-color: #BD83B8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
COLOR_MAP = {
    "Insufficient_Weight": "#60a5fa",
    "Normal_Weight": "#34d399",
    "Overweight_Level_I": "#fbbf24",
    "Overweight_Level_II": "#f97316",
    "Obesity_Type_I": "#f87171",
    "Obesity_Type_II": "#dc2626",
    "Obesity_Type_III": "#7f1d1d",
}
LABEL_ID = {
    "Insufficient_Weight": "Berat Badan Kurang",
    "Normal_Weight": "Berat Badan Normal",
    "Overweight_Level_I": "Kelebihan Berat Badan Tingkat I",
    "Overweight_Level_II": "Kelebihan Berat Badan Tingkat II",
    "Obesity_Type_I": "Obesitas Tipe I",
    "Obesity_Type_II": "Obesitas Tipe II",
    "Obesity_Type_III": "Obesitas Tipe III",
}

# Deskripsi field agar form ramah pengguna (fallback ke nama kolom asli jika tidak ada)
FIELD_INFO = {
    "Gender": ("Jenis Kelamin", "kategorik"),
    "Age": ("Usia (tahun)", "numerik"),
    "Height": ("Tinggi Badan (meter)", "numerik"),
    "Weight": ("Berat Badan (kg)", "numerik"),
    "family_history_with_overweight": ("Riwayat Keluarga Obesitas", "kategorik"),
    "FAVC": ("Sering Konsumsi Makanan Tinggi Kalori", "kategorik"),
    "FCVC": ("Frekuensi Konsumsi Sayur (1-3)", "numerik"),
    "NCP": ("Jumlah Makan Utama per Hari", "numerik"),
    "CAEC": ("Frekuensi Makan di Antara Waktu Makan", "kategorik"),
    "SMOKE": ("Merokok", "kategorik"),
    "CH2O": ("Konsumsi Air per Hari (liter)", "numerik"),
    "SCC": ("Memantau Konsumsi Kalori", "kategorik"),
    "FAF": ("Frekuensi Aktivitas Fisik (0-3)", "numerik"),
    "TUE": ("Waktu Penggunaan Perangkat Elektronik (0-2)", "numerik"),
    "CALC": ("Frekuensi Konsumsi Alkohol", "kategorik"),
    "MTRANS": ("Transportasi yang Biasa Digunakan", "kategorik"),
}


@st.cache_data
def load_dataset_from_path(path):
    df = pd.read_csv(path)
    if "BMI" not in df.columns and "Weight" in df.columns and "Height" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df


def find_local_dataset():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
    candidates = [c for c in candidates if c.lower() != "model_metrics.csv"]
    return os.path.join(base, candidates[0]) if candidates else None


# ------------------------------------------------------------------
# LOAD ARTIFACT MODEL (cache supaya tidak load ulang setiap interaksi)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))

    def p(name):
        return os.path.join(base, name)

    xgb_model = joblib.load(p("xgb_model.pkl"))
    scaler = joblib.load(p("scaler.pkl"))
    encoders = joblib.load(p("encoders.pkl"))
    le_target = joblib.load(p("label_encoder_target.pkl"))
    feature_cols = joblib.load(p("feature_cols.pkl"))

    ann_model = None
    if os.path.exists(p("ann_model.keras")):
        try:
            from tensorflow import keras
            ann_model = keras.models.load_model(p("ann_model.keras"))
        except Exception:
            ann_model = None

    metrics_df = None
    if os.path.exists(p("model_metrics.csv")):
        metrics_df = pd.read_csv(p("model_metrics.csv"))

    return xgb_model, scaler, encoders, le_target, feature_cols, ann_model, metrics_df


try:
    (xgb_model, scaler, encoders, le_target,
     feature_cols, ann_model, metrics_df) = load_artifacts()
    LOAD_ERROR = None
except Exception as e:
    LOAD_ERROR = str(e)


# ------------------------------------------------------------------
# SIDEBAR — NAVIGASI
# ------------------------------------------------------------------
st.markdown("""
<style>
/* Nav button = card rounded dengan icon di kiri */
section[data-testid="stSidebar"] div.stButton {
    margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    background: rgba(255,255,255,0.10) !important;
    color: #F5D7DB !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 10px 16px !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    box-shadow: 0 2px 6px rgba(6,20,46,0.18) !important;
    transition: background 0.2s ease !important;
    height: 48px !important;
    line-height: 1.3 !important;
    justify-content: flex-start !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(6,20,46,0.28) !important;
}
section[data-testid="stSidebar"] div.stButton > button:focus,
section[data-testid="stSidebar"] div.stButton > button:active {
    background: rgba(241,145,109,0.30) !important;
    border-left: 4px solid #F1916D !important;
    color: white !important;
    outline: none !important;
}
section[data-testid="stSidebar"] div.stButton > button p {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] div.stButton > button div {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
}
.nav-title {
    font-weight: 700;
    font-size: 1rem;
    color: #F5D7DB !important;
    border-left: 4px solid #F1916D;
    padding-left: 10px;
    margin-bottom: 10px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
section[data-testid="stSidebar"] > div {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
    text-align:center;
    margin-top:-40px;
    margin-bottom:10px;
">

<h1 style="
    margin:0;
    padding:0;
    font-size:2.4rem;
    font-weight:900;
    line-height:0.82;
">

✦
<br>

<span style="color:#BD83B8;">
MENU
</span>

<br>

<span style="color:#F1916D;">
DASHBOARD
</span>

</h1>

<p style="
    margin-top:10px;
    color:#F5D7DB;
    font-size:1rem;
    font-weight:500;
">
Obesity Prediction System
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<hr style="
border:0;
border-top:1px solid #473E66;
margin-top:15px;
margin-bottom:20px;
">
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "🔮 Prediksi"

menu_items = [
    ("⟐ Panduan",            "☰", "Panduan"),
    ("🔮 Prediksi",           "⌕", "Prediksi"),
    ("📈 EDA & Insight Data", "△", "EDA & Insight Data"),
    ("📊 Perbandingan Model", "⇄", "Perbandingan Model"),
    ("ℹ️ Tentang",            "ⓘ", "Tentang"),
]

st.sidebar.markdown("<div class='nav-title'>Navigasi</div>", unsafe_allow_html=True)

for key, icon, label in menu_items:
    # Icon + label sekaligus dalam 1 teks tombol — pasti presisi
    btn_label = f"{icon}  {label}"
    if st.sidebar.button(btn_label, key=f"nav_{key}", use_container_width=True):
        st.session_state.page = key
        st.rerun()

page = st.session_state.page

st.sidebar.markdown("<hr style='border-color:rgba(46,16,101,0.25);'>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='sidebar-caption'>💡 Dashboard ini memprediksi tingkat obesitas "
    "seseorang berdasarkan data gaya hidup dan fisik, menggunakan model "
    "<b>XGBoost</b> dan <b>ANN (Artificial Neural Network)</b>",
    unsafe_allow_html=True,
)

if LOAD_ERROR:
    st.error(
        "❌ Gagal memuat file model. Pastikan file berikut berada di folder "
        "yang sama dengan app.py: xgb_model.pkl, scaler.pkl, encoders.pkl, "
        "label_encoder_target.pkl, feature_cols.pkl.\n\nDetail error: "
        f"{LOAD_ERROR}"
    )
    st.stop()

# ------------------------------------------------------------------
# HALAMAN 1 — PREDIKSI
# ------------------------------------------------------------------
if page == "🔮 Prediksi" or page == "Prediksi":
    st.markdown(
        "<div class='hero-banner'><h1>🔮 Prediksi Tingkat Obesitas</h1>"
        "<p>Masukkan data fisik dan gaya hidup di bawah ini, lalu klik "
        "<b>Prediksi</b> untuk melihat estimasi tingkat obesitas.</p></div>",
        unsafe_allow_html=True,
    )

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        cols_cycle = [col1, col2]
        user_input = {}

        for i, col_name in enumerate(feature_cols):
            target_col = cols_cycle[i % 2]
            label, ftype = FIELD_INFO.get(col_name, (col_name, "numerik"))

            with target_col:
                if col_name in encoders:
                    classes = list(encoders[col_name].classes_)
                    user_input[col_name] = st.selectbox(label, classes, key=col_name)
                elif col_name == "Age":
                    user_input[col_name] = st.number_input(
                        label, min_value=1, max_value=120, value=25, step=1, key=col_name
                    )
                elif col_name == "Height":
                    user_input[col_name] = st.number_input(
                        label, min_value=1.0, max_value=2.5, value=1.70,
                        step=0.01, format="%.2f", key=col_name
                    )
                elif col_name == "Weight":
                    user_input[col_name] = st.number_input(
                        label, min_value=20.0, max_value=300.0, value=70.0,
                        step=0.5, key=col_name
                    )
                elif col_name in ("FCVC", "NCP", "CH2O", "FAF", "TUE"):
                    user_input[col_name] = st.slider(
                        label, min_value=0.0, max_value=3.0, value=1.0,
                        step=0.1, key=col_name
                    )
                else:
                    user_input[col_name] = st.number_input(label, value=0.0, key=col_name)

        model_choice_options = ["XGBoost"]
        if ann_model is not None:
            model_choice_options.append("ANN")
            model_choice_options.append("XGBoost & ANN (bandingkan)")

        model_choice = st.selectbox("Pilih model prediksi", model_choice_options)

        submitted = st.form_submit_button("🔎 Prediksi", use_container_width=True)

    if submitted:
        # Susun dataframe sesuai urutan feature_cols
        row = {}
        for col_name in feature_cols:
            val = user_input[col_name]
            if col_name in encoders:
                val = encoders[col_name].transform([val])[0]
            row[col_name] = val

        X_input = pd.DataFrame([row])[feature_cols]

        bmi = user_input.get("Weight", np.nan) / (user_input.get("Height", 1) ** 2)

        def predict_xgb():
            pred = xgb_model.predict(X_input)[0]
            proba = xgb_model.predict_proba(X_input)[0]
            return pred, proba

        def predict_ann():
            X_scaled = scaler.transform(X_input)
            proba = ann_model.predict(X_scaled, verbose=0)[0]
            pred = int(np.argmax(proba))
            return pred, proba

        st.markdown("---")
        st.subheader("📌 Hasil Prediksi")

        info_cols = st.columns(3)
        info_cols[0].metric("BMI (Body Mass Index)", f"{bmi:.2f}")
        info_cols[1].metric("Tinggi", f"{user_input.get('Height', 0):.2f} m")
        info_cols[2].metric("Berat", f"{user_input.get('Weight', 0):.1f} kg")

        results_to_show = []
        if model_choice in ("XGBoost", "XGBoost & ANN (bandingkan)"):
            results_to_show.append(("XGBoost", *predict_xgb()))
        if model_choice in ("ANN", "XGBoost & ANN (bandingkan)") and ann_model is not None:
            results_to_show.append(("ANN", *predict_ann()))

        result_cols = st.columns(len(results_to_show))
        for c, (model_name, pred_idx, proba) in zip(result_cols, results_to_show):
            pred_label = le_target.inverse_transform([pred_idx])[0]
            with c:
                st.markdown(f"#### Model: {model_name}")
                st.markdown(
                    f"<div style='padding:16px;border-radius:10px;"
                    f"background-color:{COLOR_MAP.get(pred_label, '#999')}22;"
                    f"border:2px solid {COLOR_MAP.get(pred_label, '#999')}'>"
                    f"<h3 style='margin:0;color:{COLOR_MAP.get(pred_label, '#999')}'>"
                    f"{LABEL_ID.get(pred_label, pred_label)}</h3></div>",
                    unsafe_allow_html=True,
                )

                proba_df = pd.DataFrame({
                    "Kelas": [LABEL_ID.get(c_, c_) for c_ in le_target.classes_],
                    "Probabilitas": proba,
                }).sort_values("Probabilitas", ascending=True)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                bar_colors = [COLOR_MAP.get(c_, "#999") for c_ in
                              sorted(le_target.classes_,
                                     key=lambda x: list(le_target.classes_).index(x))]
                ax.barh(proba_df["Kelas"], proba_df["Probabilitas"], color="#6366f1")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probabilitas")
                ax.set_title(f"Distribusi Probabilitas — {model_name}", fontsize=10)
                for y_pos, val in enumerate(proba_df["Probabilitas"]):
                    ax.text(val + 0.01, y_pos, f"{val*100:.1f}%", va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)


# ------------------------------------------------------------------
# HALAMAN 2 — PERBANDINGAN MODEL
# ------------------------------------------------------------------
elif page == "📈 EDA & Insight Data" or page == "EDA & Insight Data":
    st.markdown(
        "<div class='hero-banner'><h1>📈 EDA & Insight Data</h1>"
        "<p>Eksplorasi data yang digunakan untuk melatih model — distribusi kelas, "
        "fitur numerik, fitur kategorik, dan korelasi antar variabel.</p></div>",
        unsafe_allow_html=True,
    )

    local_csv = find_local_dataset()
    uploaded = st.file_uploader(
        "Atau upload dataset CSV (kolom harus sama dengan dataset training)",
        type=["csv"],
    )

    df_eda = None
    if uploaded is not None:
        df_eda = load_dataset_from_path(uploaded)
    elif local_csv is not None:
        df_eda = load_dataset_from_path(local_csv)
        st.caption(f"📂 Memakai dataset lokal: `{os.path.basename(local_csv)}`")
    else:
        st.info(
            "Tidak ada file CSV dataset di folder dashboard. Letakkan file dataset "
            "training (misalnya `ObesityDataSet.csv`) di folder yang sama dengan "
            "`app.py`, atau upload manual di atas, untuk menampilkan EDA."
        )

    if df_eda is not None and "NObeyesdad" in df_eda.columns:
        n_rows, n_cols = df_eda.shape
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Baris", f"{n_rows:,}")
        c2.metric("Jumlah Kolom", n_cols)
        c3.metric("Jumlah Kelas Target", df_eda["NObeyesdad"].nunique())
        c4.metric("Missing Values", int(df_eda.isnull().sum().sum()))

        st.markdown("#### 🔎 Pratinjau Data")
        st.dataframe(df_eda.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Distribusi Kelas Target")
        counts = df_eda["NObeyesdad"].value_counts().reindex(
            [c for c in ORDER if c in df_eda["NObeyesdad"].unique()]
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bar_colors = [COLOR_MAP.get(c, "#a78bfa") for c in counts.index]
        axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
        axes[0].set_title("Jumlah per Kelas", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=35)
        axes[1].pie(counts.values, labels=counts.index, colors=bar_colors,
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 7})
        axes[1].set_title("Proporsi Kelas (%)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📊 Distribusi Fitur Numerik")
        num_candidates = ["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        num_cols = [c for c in num_candidates if c in df_eda.columns]
        if num_cols:
            selected_num = st.multiselect(
                "Pilih fitur numerik untuk ditampilkan", num_cols, default=num_cols[:4]
            )
            if selected_num:
                n = len(selected_num)
                fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
                if n == 1:
                    axes = [axes]
                for ax, col in zip(axes, selected_num):
                    ax.hist(df_eda[col].dropna(), bins=20, color="#a78bfa",
                            edgecolor="white", alpha=0.85)
                    mean_val = df_eda[col].mean()
                    ax.axvline(mean_val, color="#5b21b6", linestyle="--", lw=1.2,
                               label=f"Mean={mean_val:.1f}")
                    ax.set_title(col, fontweight="bold")
                    ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 📦 Boxplot Fitur per Kelas Obesitas")
        if num_cols:
            box_col = st.selectbox("Pilih fitur untuk boxplot", num_cols, key="box_feature")
            classes_present = [c for c in ORDER if c in df_eda["NObeyesdad"].unique()]
            data = [df_eda[df_eda["NObeyesdad"] == c][box_col].dropna().values for c in classes_present]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            bp = ax.boxplot(
                data,
                patch_artist=True,
                tick_labels=classes_present
                )
            for patch, c in zip(bp["boxes"], classes_present):
                patch.set_facecolor(COLOR_MAP.get(c, "#a78bfa"))
                patch.set_alpha(0.75)
            ax.set_title(f"Distribusi {box_col} per Kelas", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 🧩 Variabel Kategorik vs Target")
        cat_candidates = ["Gender", "family_history_with_overweight", "FAVC",
                           "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
        cat_cols_present = [c for c in cat_candidates if c in df_eda.columns]
        if cat_cols_present:
            cat_choice = st.selectbox("Pilih variabel kategorik", cat_cols_present)
            ct = pd.crosstab(df_eda[cat_choice], df_eda["NObeyesdad"])
            ct = ct.reindex(columns=[c for c in ORDER if c in ct.columns], fill_value=0)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ct.plot(kind="bar", ax=ax, color=[COLOR_MAP.get(c, "#a78bfa") for c in ct.columns],
                    edgecolor="white", rot=0)
            ax.set_title(f"{cat_choice} vs Tingkat Obesitas", fontweight="bold")
            ax.legend(fontsize=7, title="Kelas")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 🔗 Heatmap Korelasi Variabel Numerik")
        if len(num_cols) >= 2:
            corr = df_eda[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="PuRd",
                        center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
            ax.set_title("Korelasi Antar Fitur Numerik", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
    elif df_eda is not None:
        st.warning(
            "Kolom `NObeyesdad` (target) tidak ditemukan di dataset yang diupload. "
            "Pastikan dataset sesuai dengan struktur data training."
        )


elif page == "📊 Perbandingan Model" or page == "Perbandingan Model":
    st.markdown(
        "<div class='hero-banner'><h1>📊 Perbandingan Performa Model</h1>"
        "<p>Lihat dan bandingkan metrik evaluasi antar model yang sudah dilatih.</p></div>",
        unsafe_allow_html=True,
    )

    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)

        metric_cols = [c for c in metrics_df.columns if c not in ("Model", "CV Std")]
        st.markdown("#### Visualisasi Perbandingan Metrik")

        def to_float(v):
            if isinstance(v, str):
                return float(v.replace("%", "").replace("±", ""))
            return float(v)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(metric_cols))
        width = 0.35
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            vals = [to_float(row[m]) for m in metric_cols]
            vals = [v / 100 if v > 1.5 else v for v in vals]
            ax.bar(x + i * width, vals, width, label=row["Model"])
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_cols, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Perbandingan Metrik Evaluasi Antar Model", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info(
            "File `model_metrics.csv` tidak ditemukan. Letakkan file tersebut "
            "di folder yang sama dengan app.py untuk menampilkan perbandingan "
            "metrik antar model (XGBoost vs ANN)."
        )

# ------------------------------------------------------------------
# HALAMAN — PANDUAN
# ------------------------------------------------------------------
elif page == "⟐ Panduan" or page == "Panduan":
    st.markdown(
        "<div class='hero-banner'><h1>⟐ Panduan Penggunaan Dashboard</h1>"
        "<p>Cara menggunakan dashboard ini langkah demi langkah.</p></div>",
        unsafe_allow_html=True,
    )

    guide_cards = [
        {
            "no": "1", "title": "Halaman Prediksi", "icon": "🔮",
            "color": "#7c3aed",
            "desc": "Isi data fisik & gaya hidup, pilih model, klik Prediksi untuk melihat hasil estimasi.",
        },
        {
            "no": "2", "title": "EDA & Insight Data", "icon": "📈",
            "color": "#0ea5e9",
            "desc": "Lihat eksplorasi data training, atau upload dataset CSV sendiri untuk dianalisis.",
        },
        {
            "no": "3", "title": "Perbandingan Model", "icon": "📊",
            "color": "#16a34a",
            "desc": "Lihat perbandingan metrik evaluasi (accuracy, F1-score, dll) antar model XGBoost & ANN.",
        },
        {
            "no": "4", "title": "Tentang Dashboard", "icon": "ℹ️",
            "color": "#2563eb",
            "desc": "Info lengkap proyek, pipeline model, dan daftar kelas target tingkat obesitas.",
        },
    ]

    st.markdown(
        """
        <style>
        .guide-card {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.12);
            margin-bottom: 20px;
            height: 100%;
        }
        .guide-card-top {
            padding: 18px 20px 14px 20px;
            color: white;
            position: relative;
            min-height: 90px;
        }
        .guide-card-top .num {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
        }
        .guide-card-top .title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 4px;
        }
        .guide-card-icon {
            position: absolute;
            right: 16px;
            top: 14px;
            font-size: 1.8rem;
            opacity: 0.85;
        }
        .guide-card-bottom {
            background: rgba(0,0,0,0.18);
            padding: 10px 16px;
            font-size: 0.85rem;
            color: white;
            font-weight: 600;
        }
        .guide-card-desc {
            background: white;
            padding: 10px 16px;
            font-size: 0.85rem;
            color: #374151;
            min-height: 70px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, card in zip(cols, guide_cards):
        with col:
            st.markdown(
                f"""
                <div class='guide-card'>
                    <div class='guide-card-top' style='background:{card["color"]};'>
                        <div class='num'>{card["no"]}</div>
                        <div class='title'>{card["title"]}</div>
                        <div class='guide-card-icon'>{card["icon"]}</div>
                    </div>
                    <div class='guide-card-desc'>{card["desc"]}</div>
                    <div class='guide-card-bottom' style='background:{card["color"]}cc;'>
                        Langkah {card["no"]} ➜
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# HALAMAN — TENTANG
# ------------------------------------------------------------------
elif page == "ℹ️ Tentang" or page == "Tentang":
    st.markdown(
        "<div class='hero-banner'><h1>ℹ️ Tentang Dashboard</h1>"
        "<p>Informasi lengkap mengenai proyek dan pipeline model.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .info-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        .info-header {
            background: #1d6fa5; color: white; font-weight: 700;
            padding: 10px 16px; font-size: 1rem;
        }
        .info-subheader {
            background: #d6dbe1; color: #1f2937; font-weight: 700;
            padding: 8px 16px; font-size: 0.9rem;
        }
        .info-row td {
            padding: 9px 16px; border-bottom: 1px solid #e5e7eb;
            font-size: 0.9rem; color: #1f2937; background: white;
        }
        .info-row td.label { width: 30%; font-weight: 600; color: #374151; }
        .info-row td.sep { width: 2%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def info_table(header, rows):
        html = f"<table class='info-table'><tr><td colspan='3' class='info-header'>{header}</td></tr>"
        for label, value in rows:
            html += (
                f"<tr class='info-row'>"
                f"<td class='label'>{label}</td><td class='sep'>:</td><td>{value}</td>"
                f"</tr>"
            )
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    info_table("PROFIL PROYEK", [
        ("Nama Proyek", "Dashboard Prediksi Tingkat Obesitas"),
        ("Tujuan", "Klasifikasi tingkat obesitas berdasarkan data fisik & gaya hidup"),
    ])

    st.markdown("<div class='info-subheader'>PIPELINE — EDA & PREPROCESSING</div>", unsafe_allow_html=True)
    info_table("", [
        ("1. EDA", "Analisis distribusi kelas, variabel numerik & kategorik"),
        ("2. Preprocessing", "Encoding fitur kategorik, train-test split (stratified)"),
        ("3. Balancing", "SMOTE untuk menyeimbangkan kelas minoritas"),
        ("4. Standardisasi", "Standardisasi fitur numerik"),
    ])

    st.markdown("<div class='info-subheader'>PIPELINE — MODELING & EVALUASI</div>", unsafe_allow_html=True)
    info_table("", [
        ("XGBoost", "Hyperparameter tuning (RandomizedSearchCV, 50 iterasi) + 5-Fold CV"),
        ("ANN", "4 hidden layer, Batch Normalization, Dropout, L2 Regularization"),
        ("Evaluasi", "Accuracy, F1-score (macro & weighted), Precision, Recall, ROC-AUC, Confusion Matrix"),
    ])

    st.markdown("<div class='info-subheader'>KELAS TARGET TINGKAT OBESITAS</div>", unsafe_allow_html=True)
    class_rows = [
        (f"<span style='color:{COLOR_MAP[cls]};font-weight:bold'>●</span> {LABEL_ID[cls]}", cls)
        for cls in ORDER
    ]
    info_table("", class_rows)

    st.markdown(
        "<div style='background:#fef3c7;border-left:4px solid #f59e0b;"
        "padding:12px 16px;border-radius:6px;font-size:0.85rem;color:#78350f;'>"
        "⚠️ <b>Catatan:</b> Hasil prediksi bersifat estimasi berdasarkan model statistik "
        "dan tidak menggantikan diagnosis medis profesional.</div>",
        unsafe_allow_html=True,
    )