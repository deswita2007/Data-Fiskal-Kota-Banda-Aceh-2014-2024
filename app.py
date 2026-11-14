import os
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Dashboard Fiskal Daerah", page_icon="📊", layout="wide")

# ---------- Kelompok variabel ----------
TRANSFER = ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark", "DAK_Fisik", "DAK_Nonfisik"]
FISKAL_LAIN = ["PAD", "ICT"]
BM = ["bm_pendidikan", "bm_kesehatan", "bm_perumahan", "bm_lingkungan", "bm_ekonomi",
      "bm_infrastruktur", "bm_pariwisata", "bm_perlindungan_sosial"]
OUTCOME = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]
NUMERIK = TRANSFER + FISKAL_LAIN + BM + OUTCOME


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ---------- Sidebar: sumber data ----------
st.sidebar.title("⚙️ Pengaturan")
upload = st.sidebar.file_uploader("Upload CSV (opsional)", type="csv")

if upload is not None:
    df = load_data(upload)
elif os.path.exists("dataset_fiskal.csv"):
    df = load_data("dataset_fiskal.csv")
else:
    st.warning("Upload file CSV di sidebar atau letakkan 'dataset_fiskal.csv' di folder yang sama dengan app.py.")
    st.stop()

if "Kota" in df.columns:
    kota = st.sidebar.multiselect("Kota", df["Kota"].unique(), default=list(df["Kota"].unique()))
    df = df[df["Kota"].isin(kota)]
if "Tahun" in df.columns:
    t_min, t_max = int(df["Tahun"].min()), int(df["Tahun"].max())
    if t_min < t_max:
        rentang = st.sidebar.slider("Rentang tahun", t_min, t_max, (t_min, t_max))
        df = df[df["Tahun"].between(*rentang)]

num_cols = [c for c in NUMERIK if c in df.columns]

# ---------- Header ----------
st.title("📊 Dashboard Analisis Fiskal Daerah")
st.caption("Transfer ke daerah, PAD, ICT, belanja modal per sektor, dan indikator pembangunan.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah observasi", len(df))
c2.metric("Jumlah kota", df["Kota"].nunique() if "Kota" in df else "-")
c3.metric("Rata-rata IPM", f"{df['IPM'].mean():.2f}" if "IPM" in df else "-")
c4.metric("Rata-rata kemiskinan (%)", f"{df['KEMISKINAN'].mean():.2f}" if "KEMISKINAN" in df else "-")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Data", "📈 Tren", "🔗 Korelasi", "🔍 Scatter", "🧮 Regresi OLS"]
)

# ---------- Tab 1: Data ----------
with tab1:
    st.subheader("Data")
    st.dataframe(df, use_container_width=True)
    st.subheader("Statistik deskriptif")
    st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)
    st.download_button("⬇️ Download data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "data_fiskal_filtered.csv", "text/csv")

# ---------- Tab 2: Tren ----------
with tab2:
    st.subheader("Tren antar tahun")
    pilih = st.multiselect("Pilih variabel", num_cols, default=["KEMISKINAN", "PENGANGGURAN"])
    if pilih:
        long = df.melt(id_vars=[c for c in ["Kota", "Tahun"] if c in df.columns],
                       value_vars=pilih, var_name="Variabel", value_name="Nilai")
        fig = px.line(long, x="Tahun", y="Nilai", color="Variabel",
                      line_dash="Kota" if df["Kota"].nunique() > 1 else None, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pilih minimal satu variabel.")

# ---------- Tab 3: Korelasi ----------
with tab3:
    st.subheader("Matriks korelasi (Pearson)")
    grup = st.radio("Kelompok variabel",
                    ["Semua", "Transfer vs Outcome", "Belanja Modal vs Outcome"], horizontal=True)
    if grup == "Semua":
        corr = df[num_cols].corr()
    elif grup == "Transfer vs Outcome":
        corr = df[TRANSFER + FISKAL_LAIN + OUTCOME].corr().loc[TRANSFER + FISKAL_LAIN, OUTCOME]
    else:
        corr = df[BM + OUTCOME].corr().loc[BM, OUTCOME]
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4: Scatter ----------
with tab4:
    st.subheader("Hubungan dua variabel")
    a, b = st.columns(2)
    x = a.selectbox("Sumbu X", num_cols, index=num_cols.index("DAU_Grant") if "DAU_Grant" in num_cols else 0)
    y = b.selectbox("Sumbu Y", num_cols, index=num_cols.index("IPM") if "IPM" in num_cols else 1)
    fig = px.scatter(df, x=x, y=y, hover_data=["Tahun"] if "Tahun" in df else None,
                     trendline="ols", color="Kota" if "Kota" in df else None)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Korelasi Pearson: **{df[x].corr(df[y]):.3f}**")

# ---------- Tab 5: Regresi ----------
with tab5:
    st.subheader("Regresi linear OLS")
    y_var = st.selectbox("Variabel dependen (Y)", OUTCOME, key="y_ols")
    x_opsi = [c for c in num_cols if c != y_var]
    x_vars = st.multiselect("Variabel independen (X)", x_opsi, default=["DAU_Grant"])
    if x_vars:
        data = df[[y_var] + x_vars].dropna()
        if len(data) <= len(x_vars) + 1:
            st.error("Jumlah observasi terlalu sedikit untuk jumlah variabel X. Kurangi variabel X.")
        else:
            X = sm.add_constant(data[x_vars])
            model = sm.OLS(data[y_var], X).fit()
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{model.rsquared:.3f}")
            m2.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
            m3.metric("N", int(model.nobs))
            st.text(model.summary().as_text())
            if len(data) < 30:
                st.warning("Sampel kecil (N < 30): hasil regresi kurang andal; gunakan sebagai eksplorasi saja.")
