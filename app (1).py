import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Data", "📈 Tren", "🔗 Korelasi", "🔍 Scatter", "🧮 Regresi OLS", "🏙️ Regresi Kota Banda Aceh"]
)

# ---------- Fungsi bantu interpretasi ----------
BAIK_JIKA = {"KEMISKINAN": "turun", "PENGANGGURAN": "turun", "KETIMPANGAN": "turun",
             "IPM": "naik", "PE": "naik"}


def kekuatan(r):
    a = abs(r)
    if a < 0.2:
        return "sangat lemah"
    if a < 0.4:
        return "lemah"
    if a < 0.6:
        return "sedang"
    if a < 0.8:
        return "kuat"
    return "sangat kuat"


def arah_korelasi(r):
    return "positif (searah)" if r > 0 else "negatif (berlawanan arah)"


def hitung_vif(data, x_names):
    """VIF manual: regresikan tiap X terhadap X lainnya."""
    hasil = {}
    for v in x_names:
        lain = [c for c in x_names if c != v]
        if not lain:
            continue
        A = np.column_stack([np.ones(len(data))] + [data[c].values for c in lain])
        yv = data[v].values
        beta = np.linalg.lstsq(A, yv, rcond=None)[0]
        res = yv - A @ beta
        sst = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - (res @ res) / sst if sst > 0 else 0
        hasil[v] = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
    return hasil


def interpretasi_ols(model, y_name, x_names, alpha, data):
    """Menampilkan interpretasi lengkap hasil regresi OLS untuk variabel X dan Y apa pun."""
    params, pvals = model.params, model.pvalues
    n, k = int(model.nobs), len(x_names)

    st.markdown("**1. Konstanta (intercept)**")
    st.write(
        f"Nilai konstanta sebesar **{params['const']:.3f}** (p = {pvals['const']:.4f}). Artinya, jika semua "
        f"variabel X bernilai nol, {y_name} diperkirakan sebesar {params['const']:.3f}. Konstanta lebih "
        f"berfungsi sebagai titik awal model, dan sering tidak bermakna praktis bila nilai nol berada di "
        f"luar rentang data."
    )

    st.markdown("**2. Koefisien regresi (pengaruh parsial tiap variabel)**")
    for v in x_names:
        b, p = params[v], pvals[v]
        arah = "naik" if b > 0 else "turun"
        sig = bool(p < alpha)
        lain = "dengan variabel X lainnya dianggap tetap, " if k > 1 else ""
        kalimat = (f"{lain}setiap kenaikan {v} sebesar 1 satuan diikuti {y_name} yang "
                   f"**{arah}** sebesar {abs(b):.4f} satuan.")
        kalimat = kalimat[0].upper() + kalimat[1:]
        st.write(
            f"- **{v}**: koefisien = **{b:.4f}** (p = {p:.4f}). {kalimat} "
            f"Pengaruh ini **{'signifikan' if sig else 'tidak signifikan'}** pada α = {alpha} "
            + ("(p < α), artinya terdapat cukup bukti bahwa variabel ini memengaruhi " + y_name + "."
               if sig else
               "(p ≥ α), artinya belum cukup bukti bahwa variabel ini memengaruhi " + y_name + ".")
        )

    st.markdown("**3. Koefisien determinasi (R² dan Adjusted R²)**")
    kualitas = ("sangat rendah" if model.rsquared < 0.2 else "rendah" if model.rsquared < 0.4
                else "sedang" if model.rsquared < 0.6 else "tinggi" if model.rsquared < 0.8 else "sangat tinggi")
    st.write(
        f"R² = **{model.rsquared:.4f}**, artinya **{model.rsquared * 100:.2f}%** variasi {y_name} dapat "
        f"dijelaskan oleh variabel X dalam model (daya jelaskan tergolong **{kualitas}**), sedangkan "
        f"**{(1 - model.rsquared) * 100:.2f}%** dijelaskan faktor lain di luar model. Adjusted R² = "
        f"{model.rsquared_adj:.4f} adalah R² yang sudah disesuaikan dengan jumlah variabel dan observasi, "
        f"sehingga lebih tepat untuk menilai kecocokan model."
    )

    st.markdown("**4. Uji simultan (uji F)**")
    f_sig = bool(model.f_pvalue < alpha)
    st.write(
        f"F-hitung = {model.fvalue:.3f} dengan p = {model.f_pvalue:.4f}. Secara bersama-sama, variabel X "
        f"**{'berpengaruh signifikan' if f_sig else 'tidak berpengaruh signifikan'}** terhadap {y_name} pada "
        f"α = {alpha}. "
        + ("Model layak digunakan untuk menjelaskan " + y_name + "." if f_sig
           else "Model secara keseluruhan belum terbukti layak menjelaskan " + y_name + ".")
    )

    nomor = 5
    if k > 1:
        st.markdown(f"**{nomor}. Variabel paling dominan (koefisien terstandar)**")
        sd_y = data[y_name].std()
        beta_std = {v: params[v] * data[v].std() / sd_y for v in x_names}
        urut = sorted(beta_std.items(), key=lambda t: abs(t[1]), reverse=True)
        daftar = ", ".join(f"{v} ({bs:+.3f})" for v, bs in urut)
        st.write(
            f"Dengan skala yang disamakan, urutan besarnya pengaruh terhadap {y_name} adalah: {daftar}. "
            f"Variabel yang paling dominan adalah **{urut[0][0]}**."
        )
        nomor += 1

        st.markdown(f"**{nomor}. Pemeriksaan multikolinearitas (VIF)**")
        vif = hitung_vif(data, x_names)
        tinggi = [v for v, val in vif.items() if val > 10]
        sedang = [v for v, val in vif.items() if 5 < val <= 10]
        vif_txt = ", ".join(f"{v} = {('∞' if np.isinf(val) else f'{val:.2f}')}" for v, val in vif.items())
        st.write(f"Nilai VIF: {vif_txt}.")
        if tinggi:
            st.warning(f"VIF > 10 pada {', '.join(tinggi)}: terindikasi multikolinearitas serius, sehingga "
                       f"koefisien dan p-value variabel tersebut kurang dapat dipercaya. Pertimbangkan "
                       f"mengurangi variabel X yang saling berkorelasi tinggi.")
        elif sedang:
            st.info(f"VIF antara 5 dan 10 pada {', '.join(sedang)}: ada indikasi korelasi antar-X yang cukup "
                    f"tinggi, interpretasikan dengan hati-hati.")
        else:
            st.write("Semua VIF < 5, tidak ada indikasi multikolinearitas yang mengganggu.")
        nomor += 1

    st.markdown(f"**{nomor}. Kesimpulan**")
    sig_vars = [v for v in x_names if pvals[v] < alpha]
    if sig_vars:
        txt = f"Variabel yang berpengaruh signifikan terhadap {y_name}: **{', '.join(sig_vars)}**."
    else:
        txt = f"Tidak ada variabel X yang berpengaruh signifikan terhadap {y_name} secara parsial."
    st.success(txt + f" Model menjelaskan {model.rsquared * 100:.2f}% variasi {y_name} (N = {n}).")
    st.caption(
        "Catatan: interpretasi dibuat otomatis dari hasil estimasi. Dengan sampel kecil dan tanpa uji asumsi "
        "klasik (normalitas, heteroskedastisitas, autokorelasi), kesimpulan sebaiknya dipandang sebagai "
        "temuan awal, bukan hubungan sebab-akibat."
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

        st.markdown("---")
        st.subheader("📝 Interpretasi Tren")
        rata = df.groupby("Tahun")[pilih].mean().sort_index()
        if "Kota" in df.columns and df["Kota"].nunique() > 1:
            st.caption("Karena ada lebih dari satu kota, nilai yang diinterpretasikan adalah rata-rata antar kota.")
        if len(rata) < 2:
            st.info("Interpretasi tren membutuhkan minimal 2 tahun data.")
        else:
            for v in pilih:
                s = rata[v].dropna()
                if len(s) < 2:
                    continue
                t_awal, t_akhir = int(s.index[0]), int(s.index[-1])
                awal, akhir = s.iloc[0], s.iloc[-1]
                selisih = akhir - awal
                persen = (selisih / awal * 100) if awal != 0 else float("nan")
                slope = np.polyfit(s.index.astype(float), s.values, 1)[0]
                rata2 = s.mean()
                cv = (s.std() / abs(rata2) * 100) if rata2 != 0 else float("nan")
                if abs(rata2) > 0 and abs(slope) / abs(rata2) < 0.01:
                    pola = "relatif stabil"
                else:
                    pola = "cenderung meningkat" if slope > 0 else "cenderung menurun"
                if cv < 10:
                    fluk = "fluktuasi rendah"
                elif cv < 25:
                    fluk = "fluktuasi sedang"
                else:
                    fluk = "fluktuasi tinggi"

                st.markdown(f"**{v}**")
                persen_txt = f", {persen:+.1f}%" if not np.isnan(persen) else ""
                st.write(
                    f"- Dari {t_awal} ke {t_akhir}, nilai berubah dari **{awal:.2f}** menjadi **{akhir:.2f}** "
                    f"(selisih {selisih:+.2f}{persen_txt}).\n"
                    f"- Secara keseluruhan tren **{pola}** (kemiringan garis tren ≈ {slope:+.3f} per tahun).\n"
                    f"- Nilai tertinggi pada **{int(s.idxmax())}** ({s.max():.2f}) dan terendah pada "
                    f"**{int(s.idxmin())}** ({s.min():.2f}); rata-rata {rata2:.2f} dengan **{fluk}** "
                    f"(koefisien variasi {cv:.1f}%)."
                )
                if v in BAIK_JIKA and pola != "relatif stabil":
                    membaik = (slope < 0) == (BAIK_JIKA[v] == "turun")
                    st.write(
                        f"- Untuk indikator {v}, kondisi yang baik adalah ketika nilainya {BAIK_JIKA[v]}, "
                        f"sehingga tren ini menunjukkan kondisi yang **{'membaik' if membaik else 'memburuk'}**."
                    )
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

    st.markdown("---")
    st.subheader("📝 Interpretasi Korelasi")
    st.markdown(
        "**Cara membaca:** nilai r berkisar −1 sampai +1. Tanda **+** berarti dua variabel bergerak searah, "
        "tanda **−** berarti berlawanan arah. Kekuatan hubungan dilihat dari |r|: "
        "< 0,2 sangat lemah; 0,2–0,4 lemah; 0,4–0,6 sedang; 0,6–0,8 kuat; ≥ 0,8 sangat kuat."
    )

    from scipy import stats as sps

    pasangan = []
    if grup == "Semua":
        kol = list(corr.columns)
        kandidat = [(kol[i], kol[j]) for i in range(len(kol)) for j in range(i + 1, len(kol))]
    else:
        kandidat = [(a_, b_) for a_ in corr.index for b_ in corr.columns]
    for a_, b_ in kandidat:
        d = df[[a_, b_]].dropna()
        if len(d) < 3 or d[a_].nunique() < 2 or d[b_].nunique() < 2:
            continue
        r, p = sps.pearsonr(d[a_], d[b_])
        pasangan.append((a_, b_, r, p))

    if not pasangan:
        st.info("Data tidak cukup untuk menghitung korelasi.")
    else:
        pasangan.sort(key=lambda t: abs(t[2]), reverse=True)
        top = pasangan[:5]
        tabel = pd.DataFrame({
            "Pasangan variabel": [f"{a_} – {b_}" for a_, b_, _, _ in top],
            "r": [round(r, 3) for _, _, r, _ in top],
            "Kekuatan": [kekuatan(r) for _, _, r, _ in top],
            "Arah": [arah_korelasi(r) for _, _, r, _ in top],
            "p-value": [round(p, 4) for _, _, _, p in top],
            "Signifikan (α=0,05)": ["Ya" if p < 0.05 else "Tidak" for _, _, _, p in top],
        })
        st.markdown("**5 hubungan terkuat pada matriks di atas**")
        st.dataframe(tabel, use_container_width=True, hide_index=True)

        for a_, b_, r, p in top[:3]:
            st.write(
                f"- **{a_}** dan **{b_}** memiliki hubungan **{kekuatan(r)}** dan **{arah_korelasi(r)}** "
                f"(r = {r:.3f}). Ketika {a_} naik, {b_} cenderung {'naik' if r > 0 else 'turun'}. "
                + ("Hubungan ini signifikan secara statistik (p < 0,05)." if p < 0.05
                   else f"Namun hubungan ini belum signifikan secara statistik (p = {p:.3f}), "
                        f"sehingga bisa saja terjadi karena kebetulan.")
            )

        n_pos = sum(1 for _, _, r, _ in pasangan if r >= 0.5)
        n_neg = sum(1 for _, _, r, _ in pasangan if r <= -0.5)
        n_sig = sum(1 for _, _, _, p in pasangan if p < 0.05)
        st.write(
            f"**Ringkasan:** dari {len(pasangan)} pasangan variabel, terdapat {n_pos} hubungan positif dan "
            f"{n_neg} hubungan negatif dengan |r| ≥ 0,5, serta {n_sig} pasangan yang signifikan pada α = 0,05."
        )
        st.caption(
            "Catatan: korelasi tidak sama dengan sebab-akibat. Dengan jumlah observasi kecil, nilai r mudah "
            "berubah dan banyak pasangan variabel bisa tampak berkorelasi hanya karena kebetulan."
        )

# ---------- Tab 4: Scatter ----------
with tab4:
    st.subheader("Hubungan dua variabel")
    a, b = st.columns(2)
    x = a.selectbox("Sumbu X", num_cols, index=num_cols.index("DAU_Grant") if "DAU_Grant" in num_cols else 0)
    y = b.selectbox("Sumbu Y", num_cols, index=num_cols.index("IPM") if "IPM" in num_cols else 1)
    fig = px.scatter(df, x=x, y=y, hover_data=["Tahun"] if "Tahun" in df else None,
                     trendline="ols", color="Kota" if "Kota" in df else None)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Interpretasi Hubungan")
    if x == y:
        st.info("Pilih dua variabel yang berbeda untuk melihat interpretasi.")
    else:
        from scipy import stats as sps

        d = df[[x, y]].dropna()
        if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
            st.info("Data tidak cukup untuk menghitung hubungan kedua variabel.")
        else:
            lr = sps.linregress(d[x], d[y])
            r, p = lr.rvalue, lr.pvalue
            sig = p < 0.05
            st.write(
                f"- **Arah dan kekuatan:** korelasi Pearson r = **{r:.3f}**, berarti hubungan {x} dan {y} "
                f"**{kekuatan(r)}** dan **{arah_korelasi(r)}**. Semakin tinggi {x}, {y} cenderung "
                f"{'semakin tinggi' if r > 0 else 'semakin rendah'}.\n"
                f"- **Garis tren:** y = {lr.intercept:.3f} + ({lr.slope:.4f})·x. Setiap kenaikan {x} sebesar "
                f"1 satuan diikuti perubahan {y} sebesar **{lr.slope:+.4f}** satuan.\n"
                f"- **Daya jelaskan:** R² = **{r ** 2:.3f}**, artinya {r ** 2 * 100:.1f}% variasi {y} "
                f"dapat dijelaskan oleh {x} saja.\n"
                f"- **Signifikansi:** p-value = {p:.4f}, hubungan ini "
                f"**{'signifikan' if sig else 'tidak signifikan'}** pada α = 0,05 (N = {len(d)})."
            )
            if sig:
                st.success(f"Kesimpulan: terdapat hubungan {kekuatan(r)} dan {arah_korelasi(r)} yang signifikan antara {x} dan {y}.")
            else:
                st.warning(f"Kesimpulan: belum ada bukti statistik yang cukup bahwa {x} berhubungan dengan {y}.")
            st.caption("Catatan: hubungan ini bersifat bivariat dan tidak membuktikan sebab-akibat; "
                       "faktor lain tidak dikendalikan.")

# ---------- Tab 5: Regresi ----------
with tab5:
    st.subheader("Regresi linear OLS")
    st.caption("Pilih Y dan X apa saja; hasil dan interpretasi menyesuaikan otomatis.")
    y_var = st.selectbox("Variabel dependen (Y)", num_cols,
                         index=num_cols.index("KEMISKINAN") if "KEMISKINAN" in num_cols else 0, key="y_ols")
    x_opsi = [c for c in num_cols if c != y_var]
    x_default = ["DAU_Grant"] if "DAU_Grant" in x_opsi else x_opsi[:1]
    x_vars = st.multiselect("Variabel independen (X)", x_opsi, default=x_default, key="x_ols")
    alpha5 = st.selectbox("Tingkat signifikansi (α)", [0.01, 0.05, 0.10], index=1, key="alpha_ols")

    if not x_vars:
        st.info("Pilih minimal satu variabel X.")
    else:
        data = df[[y_var] + x_vars].dropna()
        konstan = [v for v in x_vars if data[v].nunique() < 2]
        if konstan:
            st.warning(f"Variabel {', '.join(konstan)} bernilai konstan sehingga dikeluarkan dari model.")
            x_vars = [v for v in x_vars if v not in konstan]
            data = data[[y_var] + x_vars]

        if not x_vars:
            st.error("Semua variabel X yang dipilih bernilai konstan.")
        elif data[y_var].nunique() < 2:
            st.error(f"Variabel Y ({y_var}) bernilai konstan, regresi tidak dapat dihitung.")
        elif len(data) <= len(x_vars) + 1:
            st.error("Jumlah observasi terlalu sedikit untuk jumlah variabel X. Kurangi variabel X.")
        else:
            try:
                X = sm.add_constant(data[x_vars])
                model = sm.OLS(data[y_var], X).fit()
                if not np.all(np.isfinite(model.pvalues.values)):
                    raise ValueError("p-value tidak dapat dihitung (variabel X kemungkinan saling berkolinear sempurna).")
            except Exception as e:
                st.error(f"Regresi tidak dapat dihitung: {e}. Coba kurangi atau ganti variabel X.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{model.rsquared:.3f}")
                m2.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
                m3.metric("N", int(model.nobs))

                persamaan5 = f"Ŷ = {model.params['const']:.3f}" + "".join(
                    f" + ({model.params[v]:.3f}) {v}" for v in x_vars)
                st.markdown("**Persamaan regresi**")
                st.code(f"{persamaan5}\nŶ = {y_var}")

                with st.expander("Ringkasan hasil regresi (statsmodels)", expanded=False):
                    st.text(model.summary().as_text())

                st.markdown("---")
                st.subheader("📝 Interpretasi Hasil Regresi")
                interpretasi_ols(model, y_var, x_vars, alpha5, data)

                if len(data) < 30:
                    st.warning("Sampel kecil (N < 30): hasil regresi kurang andal; gunakan sebagai eksplorasi saja.")


# ---------- Tab 6: Regresi Kota Banda Aceh (3D) ----------
with tab6:
    st.subheader("REGRESI KOTA BANDA ACEH")
    st.caption("Regresi linear berganda: Kemiskinan = f(PAD, Pengangguran)")

    reg = pd.DataFrame({
        "Tahun": df["Tahun"].astype(int),
        "PAD": df["PAD"].astype(float),
        "Pengangguran": df["PENGANGGURAN"].astype(float),
        "Kemiskinan": df["KEMISKINAN"].astype(float),
    }).dropna()

    if len(reg) < 4:
        st.error("Data terlalu sedikit untuk regresi dengan 2 variabel X.")
    else:
        X_model = sm.add_constant(reg[["PAD", "Pengangguran"]])
        results = sm.OLS(reg["Kemiskinan"], X_model).fit()

        b0 = results.params["const"]
        b1 = results.params["PAD"]
        b2 = results.params["Pengangguran"]
        persamaan = f"Ŷ = {b0:.3f} + ({b1:.3f}) X₁ + ({b2:.3f}) X₂"

        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{results.rsquared:.4f}")
        m2.metric("Adjusted R²", f"{results.rsquared_adj:.4f}")
        m3.metric("N", int(results.nobs))

        st.markdown("**Persamaan regresi**")
        st.code(persamaan + "\nX₁ = PAD, X₂ = Pengangguran, Ŷ = Kemiskinan")

        with st.expander("Ringkasan hasil regresi (statsmodels)", expanded=False):
            st.text(results.summary().as_text())

        reg["Prediksi"] = results.predict(X_model)
        st.markdown("**Data aktual & prediksi**")
        st.dataframe(reg.round(2), use_container_width=True)

        # Grid bidang regresi
        x1 = np.linspace(reg["PAD"].min(), reg["PAD"].max(), 40)
        x2 = np.linspace(reg["Pengangguran"].min(), reg["Pengangguran"].max(), 40)
        X1, X2 = np.meshgrid(x1, x2)
        Z = b0 + b1 * X1 + b2 * X2

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=reg["PAD"], y=reg["Pengangguran"], z=reg["Kemiskinan"],
            mode="markers", marker=dict(size=7, color="red", opacity=0.9),
            name="Data Aktual", text=[f"Tahun {t}" for t in reg["Tahun"]],
            hovertemplate="<b>%{text}</b><br>PAD: %{x}<br>Pengangguran: %{y}<br>Kemiskinan: %{z}<extra></extra>",
        ))
        fig.add_trace(go.Surface(
            x=X1, y=X2, z=Z, opacity=0.60, colorscale="Viridis", name="Regression Plane",
            hovertemplate="PAD: %{x:.2f}<br>Pengangguran: %{y:.2f}<br>Prediksi Y: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title={"text": f"REGRESI LINEAR BERGANDA 3D<br><sup>{persamaan} | R² = {results.rsquared:.3f}</sup>",
                   "x": 0.5},
            scene={"xaxis": {"title": "PAD (X₁)"},
                   "yaxis": {"title": "Pengangguran (X₂)"},
                   "zaxis": {"title": "Kemiskinan (Y)"}},
            height=700, margin={"l": 0, "r": 0, "b": 0, "t": 100},
            legend={"x": 0.02, "y": 0.98},
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Interpretasi otomatis ----------
        st.markdown("---")
        st.subheader("📝 Interpretasi Hasil Regresi")
        alpha = st.selectbox("Tingkat signifikansi (α)", [0.01, 0.05, 0.10], index=1)

        def arah(b):
            return "meningkat" if b > 0 else "menurun"

        def sig_text(p):
            return ("**signifikan**" if p < alpha else "**tidak signifikan**")

        p_pad = results.pvalues["PAD"]
        p_png = results.pvalues["Pengangguran"]
        p_const = results.pvalues["const"]
        f_p = results.f_pvalue

        st.markdown("**1. Konstanta (intercept)**")
        st.write(
            f"Nilai konstanta sebesar **{b0:.3f}** (p = {p_const:.4f}). Artinya, apabila PAD dan "
            f"Pengangguran bernilai nol, Kemiskinan diperkirakan sebesar {b0:.3f}. "
            f"Konstanta ini terutama berfungsi sebagai titik awal model dan sering tidak memiliki "
            f"makna praktis jika nilai nol di luar rentang data."
        )

        st.markdown("**2. Koefisien PAD (X₁)**")
        st.write(
            f"Koefisien PAD sebesar **{b1:.3f}** (p = {p_pad:.4f}). Dengan Pengangguran dianggap tetap, "
            f"setiap kenaikan PAD sebesar 1 satuan diikuti Kemiskinan yang {arah(b1)} sebesar "
            f"{abs(b1):.3f} poin. Pengaruh ini {sig_text(p_pad)} pada α = {alpha} "
            + ("(p < α), sehingga PAD terbukti berpengaruh terhadap Kemiskinan." if p_pad < alpha
               else "(p ≥ α), sehingga belum cukup bukti bahwa PAD berpengaruh terhadap Kemiskinan.")
        )

        st.markdown("**3. Koefisien Pengangguran (X₂)**")
        st.write(
            f"Koefisien Pengangguran sebesar **{b2:.3f}** (p = {p_png:.4f}). Dengan PAD dianggap tetap, "
            f"setiap kenaikan Pengangguran sebesar 1 poin diikuti Kemiskinan yang {arah(b2)} sebesar "
            f"{abs(b2):.3f} poin. Pengaruh ini {sig_text(p_png)} pada α = {alpha} "
            + ("(p < α), sehingga Pengangguran terbukti berpengaruh terhadap Kemiskinan." if p_png < alpha
               else "(p ≥ α), sehingga belum cukup bukti bahwa Pengangguran berpengaruh terhadap Kemiskinan.")
        )

        st.markdown("**4. Koefisien determinasi (R² dan Adjusted R²)**")
        st.write(
            f"Nilai R² sebesar **{results.rsquared:.4f}**, artinya sekitar "
            f"**{results.rsquared * 100:.2f}%** variasi Kemiskinan dapat dijelaskan oleh PAD dan "
            f"Pengangguran, sedangkan **{(1 - results.rsquared) * 100:.2f}%** dijelaskan faktor lain di luar model. "
            f"Adjusted R² sebesar {results.rsquared_adj:.4f} adalah nilai R² yang sudah disesuaikan dengan "
            f"jumlah variabel dan observasi, sehingga lebih tepat dipakai untuk menilai kecocokan model."
        )

        st.markdown("**5. Uji simultan (uji F)**")
        st.write(
            f"Nilai F-hitung = {results.fvalue:.3f} dengan p = {f_p:.4f}. Secara bersama-sama, PAD dan "
            f"Pengangguran {sig_text(f_p)} memengaruhi Kemiskinan pada α = {alpha}. "
            + ("Model layak digunakan untuk menjelaskan Kemiskinan." if f_p < alpha
               else "Model secara keseluruhan belum terbukti layak menjelaskan Kemiskinan.")
        )

        st.markdown("**6. Kesimpulan**")
        sig_vars = [n for n, p in [("PAD", p_pad), ("Pengangguran", p_png)] if p < alpha]
        if sig_vars:
            kesimpulan = (f"Variabel yang berpengaruh signifikan terhadap Kemiskinan: **{', '.join(sig_vars)}**.")
        else:
            kesimpulan = "Tidak ada variabel yang berpengaruh signifikan terhadap Kemiskinan secara parsial."
        st.success(kesimpulan + f" Model menjelaskan {results.rsquared * 100:.2f}% variasi Kemiskinan.")

        st.caption(
            "Catatan: interpretasi ini dibuat otomatis dari hasil estimasi. Dengan sampel kecil dan tanpa uji "
            "asumsi klasik (normalitas, multikolinearitas, heteroskedastisitas, autokorelasi), kesimpulan "
            "sebaiknya dipandang sebagai temuan awal, bukan hubungan sebab-akibat."
        )

        if len(reg) < 30:
            st.warning("Sampel kecil (N < 30): hasil regresi bersifat eksploratif.")
