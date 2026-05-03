"""
Akilli Sebeke Komuta Merkezi - Dashboard
- Sadece Gemini AI entegrasyonu
- Gemini API key kodda gomulu
- Turkuaz AI analiz butonu
- Pasta grafikde gunes sari, fosil koyu/karanlik
- Emoji kullanilmiyor
- Anomali tespiti, ML tahmini, karbon takibi
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import timedelta, datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
import os
import io

# ─── 1. SAYFA AYARLARI ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Akilli Sebeke Komuta Merkezi",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #00BCD4 !important;
        border-color: #00BCD4 !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #0097A7 !important;
        border-color: #0097A7 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:active {
        background-color: #00838F !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── 2. GEMINI API KEY  ───────────────────────────────────────────────
GEMINI_KEY = "AIzaSyAxNuaL_bQ2ZrXsa1a_Dt_eEJL3pTT38Gs"

# ─── 3. SABITLER ──────────────────────────────────────────────────────────────
LAT, LON              = 40.77, 30.40
FOSIL_KARBON_KATSAYI  = 550

# Pasta & bar grafik renkleri
RENK_GUNES   = "#F9C931"   # sari
RENK_RUZGAR  = "#3498DB"   # mavi
RENK_FOSIL   = "#2C3E50"   # koyu lacivert/antrasit
RENK_JEOTERM = "#C0392B"   # koyu kirmizi
RENK_BARAJLI = "#1ABC9C"   # teal
RENK_BIYOK   = "#6D4C41"   # koyu kahverengi

# ─── 4. VERI KATMANI ─────────────────────────────────────────────────────────
def _sayi_duzelt(df: pd.DataFrame, kolon: str) -> pd.DataFrame:
    if kolon in df.columns:
        df[kolon] = (
            df[kolon].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return df


def _sutun_bul(df: pd.DataFrame, anahtar_kelimeler: list) -> str | None:
    """Sutun adini encoding farkindan bagimsiz bul."""
    for col in df.columns:
        norm = col.lower().replace(" ", "").translate(
            str.maketrans("üşğıöç", "usgioç".replace("ç","c"))
        )
        if all(k.lower() in col.lower() for k in anahtar_kelimeler):
            return col
    return None


@st.cache_data
def tarihi_veri_yukle() -> pd.DataFrame:
    t = pd.read_csv("tuketim.csv", sep=";")
    u = pd.read_csv("uretim.csv",  sep=";", skiprows=3)
    t.columns = t.columns.str.strip()
    u.columns = u.columns.str.strip()

    tuk_kolon = next((c for c in t.columns if "ketim" in c and "MWh" in c), None)
    if tuk_kolon is None:
        raise KeyError(f"Tuketim sutunu bulunamadi: {list(t.columns)}")

    t = _sayi_duzelt(t, tuk_kolon)
    for col in u.columns[2:]:
        u = _sayi_duzelt(u, col)

    t["zaman"] = pd.to_datetime(
        t["Tarih"].astype(str) + " " + t["Saat"].astype(str),
        dayfirst=True, errors="coerce"
    )
    u["zaman"] = pd.to_datetime(
        u["Tarih"].astype(str) + " " + u["Saat"].astype(str),
        dayfirst=True, errors="coerce"
    )

    hedef_map = {
        "Toplam": ["Toplam"],
        "Gunes":  ["ne", "MWh"],   
        "Ruzgar": ["zgar"],
        "Dogal Gaz": ["Gaz"],
        "Ithal Komur": ["thal"],
        "Linyit": ["inyit"],
        "Jeotermal": ["eotermal"],
        "Barajli": ["arajl"],
        "Akarsu": ["karsu"],
        "Biyokutle": ["iyok"],
    }
    u_cols = [c for c in u.columns if c not in ("Tarih", "Saat")]
    u_cols = list(dict.fromkeys(u_cols))  

    df = pd.merge(t[["zaman", tuk_kolon]], u[u_cols], on="zaman", how="inner")
    df.rename(columns={tuk_kolon: "Tuketim"}, inplace=True)
    df.dropna(subset=["zaman"], inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values("zaman", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["saat"]      = df["zaman"].dt.hour
    df["gun_hafta"] = df["zaman"].dt.dayofweek
    df["ay"]        = df["zaman"].dt.month
    df["haftasonu"] = (df["gun_hafta"] >= 5).astype(int)

    for lag in [24, 48, 168]:
        df[f"tuketim_lag{lag}"] = df["Tuketim"].shift(lag)

    gunes_col   = next((c for c in df.columns if "ne" in c.lower() and c not in ["zaman","Tuketim","Toplam"]), None)
    ruzgar_col  = next((c for c in df.columns if "zgar" in c.lower()), None)
    dg_col      = next((c for c in df.columns if "Gaz" in c), None)
    ik_col      = next((c for c in df.columns if "thal" in c), None)
    lin_col     = next((c for c in df.columns if "inyit" in c), None)

    df["Gunes_col"]  = gunes_col  or ""
    df["Ruzgar_col"] = ruzgar_col or ""

    g  = df[gunes_col]  if gunes_col  else pd.Series(0, index=df.index)
    r  = df[ruzgar_col] if ruzgar_col else pd.Series(0, index=df.index)
    dg = df[dg_col]     if dg_col     else pd.Series(0, index=df.index)
    ik = df[ik_col]     if ik_col     else pd.Series(0, index=df.index)
    ln = df[lin_col]    if lin_col    else pd.Series(0, index=df.index)

    toplam_col = next((c for c in df.columns if c == "Toplam"), None)
    toplam_s   = df[toplam_col] if toplam_col else df["Tuketim"]

    df["Temiz_Enerji"] = g + r
    df["Fosil_Enerji"] = dg + ik + ln
    df["Yesil_Oran"]   = np.where(toplam_s > 0, df["Temiz_Enerji"] / toplam_s * 100, 0)
    df["Karbon_gCO2_kWh"] = np.where(toplam_s > 0, df["Fosil_Enerji"] / toplam_s * FOSIL_KARBON_KATSAYI, 0)

    df["gunes_lag24"]  = g.shift(24)
    df["ruzgar_lag24"] = r.shift(24)
    df["gunes_lag168"] = g.shift(168)
    df["ruzgar_lag168"]= r.shift(168)
    df["tuketim_lag168"]= df["Tuketim"].shift(168)

    return df.dropna()


def canli_veri_getir() -> pd.DataFrame:
    dosya = "canli_veri.csv"
    if not os.path.exists(dosya):
        st.warning("`canli_veri.csv` bulunamadi. Terminalde `python bot.py` ile botu baslatın.")
        st.stop()
    df = pd.read_csv(dosya)
    df["zaman"] = pd.to_datetime(df["zaman"], format="mixed")
    df.fillna(0, inplace=True)

    tuk = next((c for c in df.columns if "ketim" in c and "MWh" in c), None)
    if tuk and tuk != "Tuketim":
        df.rename(columns={tuk: "Tuketim"}, inplace=True)

    gunes_col  = next((c for c in df.columns if "ne" in c.lower() and c not in ["zaman","Tuketim","Toplam"] and "lag" not in c), None)
    ruzgar_col = next((c for c in df.columns if "zgar" in c.lower() and "lag" not in c), None)

    if "Temiz_Enerji" not in df.columns:
        g = df[gunes_col]  if gunes_col  else pd.Series(0, index=df.index)
        r = df[ruzgar_col] if ruzgar_col else pd.Series(0, index=df.index)
        df["Temiz_Enerji"] = g + r

    toplam_col = next((c for c in df.columns if c == "Toplam"), None)
    toplam_s   = df[toplam_col] if toplam_col else df.get("Tuketim", pd.Series(0, index=df.index))

    if "Yesil_Oran" not in df.columns:
        df["Yesil_Oran"] = np.where(toplam_s > 0, df["Temiz_Enerji"] / toplam_s * 100, 0)

    if "Karbon_gCO2_kWh" not in df.columns:
        fosil = df.get("Fosil_Enerji", pd.Series(0, index=df.index))
        df["Karbon_gCO2_kWh"] = np.where(toplam_s > 0, fosil / toplam_s * FOSIL_KARBON_KATSAYI, 0)

    df["_gunes"]  = df[gunes_col]  if gunes_col  else 0
    df["_ruzgar"] = df[ruzgar_col] if ruzgar_col else 0

    return df


@st.cache_resource
def modelleri_egit(df: pd.DataFrame):
    ozellikler = [
        "saat", "gun_hafta", "ay", "haftasonu",
        "tuketim_lag24", "gunes_lag24", "ruzgar_lag24",
        "tuketim_lag168", "gunes_lag168", "ruzgar_lag168"
    ]
    X = df[ozellikler]

    def pipe():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42
            ))
        ])

    gunes_col_ad  = next((c for c in df.columns if "ne" in c.lower() and c not in ["zaman","Tuketim","Toplam"] and "lag" not in c and "col" not in c), None)
    ruzgar_col_ad = next((c for c in df.columns if "zgar" in c.lower() and "lag" not in c and "col" not in c), None)
    gunes_hedef  = df[gunes_col_ad].clip(lower=0).fillna(0)  if gunes_col_ad  else df["Temiz_Enerji"].fillna(0)
    ruzgar_hedef = df[ruzgar_col_ad].clip(lower=0).fillna(0) if ruzgar_col_ad else pd.Series(0, index=df.index)
    mt = pipe().fit(X, df["Tuketim"].clip(lower=0))
    mg = pipe().fit(X, gunes_hedef)
    mr = pipe().fit(X, ruzgar_hedef)
    return mt, mg, mr, ozellikler


@st.cache_data(ttl=600)
def hava_durumu_cek() -> pd.DataFrame:
    dosya = "hava_durumu_veri.csv"
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,windspeed_10m,cloudcover,precipitation_probability"
        f"&timezone=Europe%2FIstanbul&forecast_days=3"
    )
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            h = r.json()["hourly"]
            df_h = pd.DataFrame({
                "tarih_saat": pd.to_datetime(h["time"]),
                "sicaklik":   h["temperature_2m"],
                "ruzgar":     h["windspeed_10m"],
                "bulut":      h["cloudcover"],
                "yagis_olas": h.get("precipitation_probability", [0] * len(h["time"]))
            })
            df_h.to_csv(dosya, index=False)
    except Exception:
        pass
    if os.path.exists(dosya):
        df_k = pd.read_csv(dosya)
        df_k["tarih_saat"] = pd.to_datetime(df_k["tarih_saat"])
        return df_k
    now = datetime.now()
    return pd.DataFrame({
        "tarih_saat": [now + timedelta(hours=i) for i in range(72)],
        "sicaklik":   [20] * 72,
        "ruzgar":     [10] * 72,
        "bulut":      [50] * 72,
        "yagis_olas": [0]  * 72
    })


# ─── 5. ANOMALI TESPITI ────────────────────────────────────────────────────────
def anomali_tespit(seri: pd.Series, esik: float = 2.5) -> pd.Series:
    z = (seri - seri.mean()) / (seri.std() + 1e-9)
    return z.abs() > esik


# ─── 6. GEMINI ANALIZI ────────────────────────────────────────────────────────
def gemini_analiz(prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        modeller = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods and "gemini" in m.name
        ]
        if not modeller:
            return "Kullanilabilir Gemini modeli bulunamadi."
        model = genai.GenerativeModel(modeller[0])
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini API hatasi: {e}"


# ─── 7. SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("Kontrol Paneli")

canli_aktif = st.sidebar.checkbox("Otomatik Yenileme (3 sn)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Senaryo Simülasyonu")
st.sidebar.caption("Hava koşullarini değiştirerek şebeke tepkisini modelleyin.")
sim_sicaklik = st.sidebar.slider("Sıcaklık (C)",       -15, 15, 0)
sim_ruzgar   = st.sidebar.slider("Rüzgar (km/h)",      -20, 50, 0)
sim_bulut    = st.sidebar.slider("Bulutluluk (%)",      -50, 50, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Gösterim Penceresi")
geri_pencere = st.sidebar.slider("Son kaç saat gösterilsin?", 6, 168, 24)

# ─── 8. VERI YUKLEMESI ────────────────────────────────────────────────────────
df_tarihi = tarihi_veri_yukle()
df_canli  = canli_veri_getir()
mt, mg, mr, ozellikler = modelleri_egit(df_tarihi)
df_hava   = hava_durumu_cek()

son_tarih   = df_canli["zaman"].max()
yarin_zaman = [son_tarih + timedelta(hours=i) for i in range(1, 25)]

hava_slice = df_hava[df_hava["tarih_saat"] > son_tarih].head(24)
if len(hava_slice) < 24:
    hava_slice = pd.concat([hava_slice] * (24 // max(len(hava_slice), 1) + 1)).head(24)

sicakliklar = hava_slice["sicaklik"].tolist()
ruzgar_hava = hava_slice["ruzgar"].tolist()
bulut_hava  = hava_slice["bulut"].tolist()

# ─── 9. TAHMIN ─────────────────────────────────────────────────────────────────
def guvenli_lag(seri: pd.Series, pencere: int, hedef_uzunluk: int = 24) -> np.ndarray:
    """Son hedef_uzunluk saatin, pencere saat oncesindeki degerlerini dondurur."""
    vals = seri.values
    if len(vals) >= pencere + hedef_uzunluk:
        segment = vals[-(pencere + hedef_uzunluk):-pencere]
    elif len(vals) >= pencere:
        segment = vals[:-pencere]
    else:
        segment = vals
        
    if len(segment) < hedef_uzunluk:
        fark = hedef_uzunluk - len(segment)
        if len(segment) > 0:
            # Veri varsa son degeri uzat (sabit deger gondermeden)
            segment = np.pad(segment, (fark, 0), mode="edge")
        else:
            # Hic veri yoksa sifir bas
            segment = np.pad(segment, (fark, 0), mode="constant", constant_values=0.0)
            
    return segment[-hedef_uzunluk:]


yarin_X = pd.DataFrame({
    "saat":           [t.hour       for t in yarin_zaman],
    "gun_hafta":      [t.dayofweek  for t in yarin_zaman],
    "ay":             [t.month      for t in yarin_zaman],
    "haftasonu":      [1 if t.dayofweek >= 5 else 0 for t in yarin_zaman],
    "tuketim_lag24":  guvenli_lag(df_canli["Tuketim"],  24),
    "gunes_lag24":    guvenli_lag(df_canli["_gunes"],   24),
    "ruzgar_lag24":   guvenli_lag(df_canli["_ruzgar"],  24),
    "tuketim_lag168": guvenli_lag(df_canli["Tuketim"], 168),
    "gunes_lag168":   guvenli_lag(df_canli["_gunes"],  168),
    "ruzgar_lag168":  guvenli_lag(df_canli["_ruzgar"], 168),
})

ham_tuketim = mt.predict(yarin_X)
ham_gunes   = mg.predict(yarin_X)
ham_ruzgar  = mr.predict(yarin_X)

_t_min = df_tarihi["Tuketim"].quantile(0.01)
_t_max = df_tarihi["Tuketim"].quantile(0.99) * 1.20
_g_max = df_tarihi["Temiz_Enerji"].quantile(0.99) * 1.20
ham_tuketim = np.clip(ham_tuketim, _t_min, _t_max)
ham_gunes   = np.clip(ham_gunes,   0,      _g_max)
ham_ruzgar  = np.clip(ham_ruzgar,  0,      _g_max)

tahmin_tuketim, tahmin_gunes, tahmin_ruzgar = [], [], []
for i in range(24):
    t_sic = sicakliklar[i] + sim_sicaklik
    t_ruz = max(0, ruzgar_hava[i] + sim_ruzgar)
    t_bul = max(0, min(100, bulut_hava[i] + sim_bulut))
    saat  = yarin_zaman[i].hour

    tt = ham_tuketim[i]
    tg = ham_gunes[i]
    tr = ham_ruzgar[i]

    if t_sic < 12:   tt *= 1 + (12 - t_sic) * 0.015
    elif t_sic > 28: tt *= 1 + (t_sic - 28) * 0.020

    if t_ruz > 15:   tr *= 1 + (t_ruz - 15) * 0.015

    if saat < 6 or saat > 19:
        tg = 0
    else:
        if t_bul > 60:   tg *= 1 - (t_bul - 60) / 100
        elif t_bul < 30: tg *= 1 + (30 - t_bul) / 100

    tahmin_tuketim.append(max(0, tt))
    tahmin_gunes.append(max(0, tg))
    tahmin_ruzgar.append(max(0, tr))

tahmin_yesil = [g + r for g, r in zip(tahmin_gunes, tahmin_ruzgar)]
tahmin_fosil = [max(0, t - y) for t, y in zip(tahmin_tuketim, tahmin_yesil)]

# ─── 10. ANOMALI & ESIK ───────────────────────────────────────────────────────
son_n            = df_canli.tail(geri_pencere)
anomali_maske    = anomali_tespit(son_n["Tuketim"])
anomali_noktalar = son_n[anomali_maske]

beklenen_tepe    = max(tahmin_tuketim)
son_24_ort       = df_canli["Tuketim"].tail(24).mean()

gecmis_seri  = df_canli["Tuketim"].tail(168)
gecmis_ort   = gecmis_seri.mean()
gecmis_std   = gecmis_seri.std()
dinamik_esik = gecmis_ort + 2.5 * gecmis_std
acil_uyari   = beklenen_tepe > dinamik_esik

# ─── 11. BASLIK & UYARILAR ────────────────────────────────────────────────────
st.title("Akıllı Şebeke Komuta Merkezi")
st.markdown(
    f"Son güncelleme: **{son_tarih.strftime('%d.%m.%Y %H:00')}** | "
    f"Canlı veri: **{len(df_canli)} saat**"
)

if acil_uyari:
    asim_mwh = beklenen_tepe - dinamik_esik
    st.error(
        f"KRITIK: Beklenen tepe yük **{beklenen_tepe:,.0f} MWh** — "
        f"tarihi normalin **{asim_mwh:+,.0f} MWh** üzerinde! Frekans çökmesi riski."
    )

if len(anomali_noktalar) > 0:
    st.warning(
        f"Son {geri_pencere} saatte **{len(anomali_noktalar)} anomalik** tuketim noktas tespit edildi."
    )

# ─── 12. ANLIK GOSTERGELER ────────────────────────────────────────────────────
anlik_tuketim = df_canli["Tuketim"].iloc[-1]
anlik_yesil   = df_canli["Temiz_Enerji"].iloc[-1]
anlik_toplam  = df_canli["Toplam"].iloc[-1] if "Toplam" in df_canli.columns else anlik_tuketim
anlik_oran    = anlik_yesil / anlik_toplam * 100 if anlik_toplam > 0 else 0
anlik_karbon  = (anlik_toplam - anlik_yesil) / anlik_toplam * FOSIL_KARBON_KATSAYI if anlik_toplam > 0 else 0

st.markdown("#### Anlık Şebeke Durumu")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tüketim",          f"{anlik_tuketim:,.0f} MWh")
c2.metric("Yeşil Oran",       f"%{anlik_oran:.1f}", delta=f"%{anlik_oran - 30:.1f} hedeften")
c3.metric("Karbon Yoğunluğu", f"{anlik_karbon:.0f} gCO2/kWh")
c4.metric("Rüzgar + Günes",   f"{anlik_yesil:,.0f} MWh")

st.markdown("#### Gelecek 24 Saat Tahminleri")
c5, c6, c7, c8 = st.columns(4)
yesil_ort = (sum(tahmin_yesil) / sum(tahmin_tuketim)) * 100 if sum(tahmin_tuketim) > 0 else 0
c5.metric(
    "Tepe Yük", f"{beklenen_tepe:,.0f} MWh",
    delta=f"%{((beklenen_tepe - son_24_ort) / son_24_ort) * 100:.1f}",
    delta_color="inverse" if acil_uyari else "normal"
)
c6.metric("Maks. Güneş",     f"{max(tahmin_gunes):,.0f} MWh")
c7.metric("Maks. Rüzgar",    f"{max(tahmin_ruzgar):,.0f} MWh")
c8.metric("Ort. Yeşil Oran", f"%{yesil_ort:.1f}")

st.divider()

# ─── 13. ANA GRAFIK ───────────────────────────────────────────────────────────
st.markdown("### Tüketim ve Yenilenebilir Üretim Trendi")
fig1 = go.Figure()

gecmis = df_canli.tail(geri_pencere)
fig1.add_trace(go.Scatter(
    x=gecmis["zaman"], y=gecmis["Tuketim"],
    mode="lines", name="Gerçek Tüketim",
    line=dict(color="#95a5a6", width=2, dash="dot")
))
fig1.add_trace(go.Scatter(
    x=gecmis["zaman"], y=gecmis["Temiz_Enerji"],
    mode="lines", name="Gerçek Yeşil",
    line=dict(color="#27ae60", width=2, dash="dot")
))
if len(anomali_noktalar) > 0:
    fig1.add_trace(go.Scatter(
        x=anomali_noktalar["zaman"], y=anomali_noktalar["Tuketim"],
        mode="markers", name="Anomali",
        marker=dict(color="#E53935", size=10, symbol="x")
    ))
fig1.add_trace(go.Scatter(
    x=yarin_zaman, y=tahmin_tuketim,
    mode="lines", name="Tahmini Tüketim",
    line=dict(color="#e74c3c", width=3)
))
fig1.add_trace(go.Scatter(
    x=yarin_zaman, y=tahmin_yesil,
    mode="lines", name="Tahmini Yeşil",
    line=dict(color="#2ecc71", width=3),
    fill="tozeroy", fillcolor="rgba(46,204,113,0.15)"
))
fig1.add_shape(
    type="line",
    x0=son_tarih.isoformat(), x1=son_tarih.isoformat(),
    y0=0, y1=1, xref="x", yref="paper",
    line=dict(color="orange", dash="dash", width=1.5)
)
fig1.add_annotation(
    x=son_tarih.isoformat(), y=1,
    xref="x", yref="paper",
    text="Tahmin başlıyor", showarrow=False,
    xanchor="left", yanchor="top",
    font=dict(color="orange", size=11)
)
fig1.update_layout(
    hovermode="x unified", height=420,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Zaman", yaxis_title="MWh"
)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

# ─── 14. ALT GRAFIKLER ────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Enerji Dagilimi (24 Saat Tahmini)")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=yarin_zaman, y=tahmin_gunes,  name="Güneş",  marker_color=RENK_GUNES))
    fig2.add_trace(go.Bar(x=yarin_zaman, y=tahmin_ruzgar, name="Rüzgar", marker_color=RENK_RUZGAR))
    fig2.add_trace(go.Bar(x=yarin_zaman, y=tahmin_fosil,  name="Fosil",  marker_color=RENK_FOSIL, opacity=0.8))
    fig2.update_layout(
        barmode="stack", height=320, hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with col_r:
    st.markdown("#### Hava Durumu (Simülasyonlu)")
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    sim_sic = [s + sim_sicaklik for s in sicakliklar]
    sim_ruz = [max(0, r + sim_ruzgar) for r in ruzgar_hava]
    sim_bul = [max(0, min(100, b + sim_bulut)) for b in bulut_hava]

    fig3.add_trace(go.Scatter(
        x=yarin_zaman, y=sim_sic, name="Sıcaklık (C)",
        line=dict(color="#e67e22")
    ), secondary_y=False)
    fig3.add_trace(go.Scatter(
        x=yarin_zaman, y=sim_ruz, name="Rüzgar (km/h)",
        line=dict(color="#9b59b6")
    ), secondary_y=True)
    fig3.add_trace(go.Scatter(
        x=yarin_zaman, y=sim_bul, name="Bulut (%)",
        line=dict(color="#bdc3c7", dash="dot")
    ), secondary_y=True)
    fig3.update_layout(
        height=320, hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ─── 15. PASTA + KARBON ───────────────────────────────────────────────────────
st.divider()
st.markdown("### Günlük Enerji Payları ve Karbon İzi")

col_pie, col_karbon = st.columns([1, 2])

with col_pie:
    toplam_gunes  = max(sum(tahmin_gunes),  0.01)
    toplam_ruzgar = max(sum(tahmin_ruzgar), 0.01)
    toplam_fosil  = max(sum(tahmin_fosil),  0.01)

    fig_pie = px.pie(
        names=["Güneş", "Rüzgar", "Fosil/Doğalgaz"],
        values=[toplam_gunes, toplam_ruzgar, toplam_fosil],
        color_discrete_sequence=[
            RENK_FOSIL,   # Gunes  -> koyu antrasit (takas)
            RENK_RUZGAR,  # Ruzgar -> mavi
            RENK_GUNES,   # Fosil  -> sari (takas)
        ],
        hole=0.45
    )
    fig_pie.update_traces(
        textfont_size=13,
        marker=dict(line=dict(color="#1a1a2e", width=2))
    )
    fig_pie.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

with col_karbon:
    karbon_tahmini = [
        (f / t) * FOSIL_KARBON_KATSAYI if t > 0 else 0
        for f, t in zip(tahmin_fosil, tahmin_tuketim)
    ]
    gercek_karbon = gecmis["Karbon_gCO2_kWh"] if "Karbon_gCO2_kWh" in gecmis.columns else pd.Series([0] * len(gecmis))

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=gecmis["zaman"], y=gercek_karbon,
        mode="lines", name="Gerçek CO2",
        line=dict(color="#e74c3c", dash="dot")
    ))
    fig_k.add_trace(go.Scatter(
        x=yarin_zaman, y=karbon_tahmini,
        mode="lines", name="Tahmini CO2",
        line=dict(color="#c0392b", width=3),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.1)"
    ))
    fig_k.update_layout(
        height=320, hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="gCO2/kWh",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_k, use_container_width=True, config={"displayModeBar": False})

# ─── 16. AI ANALIZI ───────────────────────────────────────────────────────────
st.divider()
st.markdown("### Yapay Zeka Şebeke Analizi")

col_ai_btn, col_ai_out = st.columns([1, 2])

prompt_base = f"""
Sen Turkiye Akilli Elektrik Sebekesi bas muhendisisin.

Asagidaki verilere dayanarak teknik bir analiz raporu yaz.

Anlik Tuketim: {anlik_tuketim:,.0f} MWh
Anlik Yesil Oran: %{anlik_oran:.1f}
Karbon Yogunlugu: {anlik_karbon:.0f} gCO2/kWh
Yarin Beklenen Tepe Yuk: {beklenen_tepe:,.0f} MWh
Yarin Toplam Yesil Uretim: {sum(tahmin_yesil):,.0f} MWh
Yarin Toplam Tuketim: {sum(tahmin_tuketim):,.0f} MWh
Yesil Enerji Orani: %{yesil_ort:.1f}
Anomali Durumu: {"KRITIK - Tehlikeli Tepe" if acil_uyari else "Stabil"}
Tespit Edilen Anomali Sayisi: {len(anomali_noktalar)}
Simulasyon: Sicaklik {sim_sicaklik:+}C, Ruzgar {sim_ruzgar:+} km/h, Bulut {sim_bulut:+}%

ONEMLI KURALLAR:
- Rapor basligi, kurum adi, konu satiri, tarih, rapor numarasi yazma.
- "Sayın Yönetim Kurulu", "Bilgilerinize arz ederim", imza, unvan gibi resmi yazi kalıpları kullanma.
- Dogrudan analiz icerigine gir.
- Yalnizca asagidaki 3 baslik altinda yaz, baska ek bolum ekleme:

### 1. Sebeke Yuku ve Risk Analizi
### 2. Yenilenebilir Enerji ve Karbon Durumu
### 3. Otonom Aksiyon Onerileri (3 teknik madde)
"""

with col_ai_btn:
    if "ai_raporu" not in st.session_state:
        st.session_state["ai_raporu"] = ""

    if st.button("Derin Risk Analizi Baslat", type="primary"):
        with st.spinner("Gemini analiz yapiyor..."):
            st.session_state["ai_raporu"] = gemini_analiz(prompt_base)

with col_ai_out:
    if st.session_state["ai_raporu"]:
        st.info(st.session_state["ai_raporu"])
        st.caption("Motor: Gemini AI")

# ─── 17. HAM VERI TABLOSU & EXPORT ───────────────────────────────────────────
st.divider()
with st.expander("Ham Veri Tablosu ve Export"):
    goster_kolonlar = [c for c in ["zaman", "Tuketim", "Toplam", "Temiz_Enerji", "Yesil_Oran"] if c in df_canli.columns]
    st.dataframe(
        df_canli.tail(geri_pencere)[goster_kolonlar].sort_values("zaman", ascending=False),
        use_container_width=True
    )

    csv_buf = io.StringIO()
    df_canli.to_csv(csv_buf, index=False)
    st.download_button(
        label="Tum Canli Veriyi CSV Indir",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"sebeke_veri_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    df_tahmin = pd.DataFrame({
        "zaman":                  yarin_zaman,
        "tahmin_tuketim":         tahmin_tuketim,
        "tahmin_gunes":           tahmin_gunes,
        "tahmin_ruzgar":          tahmin_ruzgar,
        "tahmin_fosil":           tahmin_fosil,
        "tahmin_karbon_gCO2_kWh": karbon_tahmini,
    })
    csv_buf2 = io.StringIO()
    df_tahmin.to_csv(csv_buf2, index=False)
    st.download_button(
        label="24 Saatlik Tahminleri CSV Indir",
        data=csv_buf2.getvalue().encode("utf-8"),
        file_name=f"tahminler_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ─── 18. AUTO-REFRESH ─────────────────────────────────────────────────────────
if canli_aktif:
    time.sleep(3)
    st.rerun()
