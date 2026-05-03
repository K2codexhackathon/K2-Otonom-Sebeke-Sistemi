"""
Akilli Sebeke - Canli Veri Toplayici Bot
- Yapilandirilab ilir hiz (hackathon / demo / test / gercek)
- Graceful shutdown (Ctrl+C ile guvenli kapanis)
- Dongusel veri akisi (veri bitince basa doner)
- Detayli loglama
- Hata toleransi (tek satir hatasi botu durdurmaz)
- Istatistik ozeti
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import signal
import logging
from datetime import datetime

# --- YAPILANDIRMA ---
CANLI_DOSYA      = "canli_veri.csv"
TUKETIM_CSV      = "tuketim.csv"
URETIM_CSV       = "uretim.csv"

HIZLANDIRMA_MODU = os.environ.get("BOT_MOD", "hackathon")
MOD_SURELER = {
    "hackathon": 3,
    "demo":      1,
    "test":      0.1,
    "gercek":    3600,
}
BEKLEME_SURESI = MOD_SURELER.get(HIZLANDIRMA_MODU, 3)
ILK_PENCERE    = 24
DONGU_MU       = True

# --- LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# --- GRACEFUL SHUTDOWN ---
_calis = True

def _cikis_isleci(sig, frame):
    global _calis
    log.info("Durdurma sinyali alindi. Bot guvenle kapatiliyor...")
    _calis = False

signal.signal(signal.SIGINT,  _cikis_isleci)
signal.signal(signal.SIGTERM, _cikis_isleci)


# --- YARDIMCI: Turkce sayi formati duzeltici ---
def sayi_duzelt(df: pd.DataFrame, kolon: str) -> pd.DataFrame:
    """Turkce sayi formatini (1.234,56 -> 1234.56) duzeltir."""
    if kolon in df.columns:
        df[kolon] = (
            df[kolon].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return df


# --- VERI HAZIRLAMA ---
def veri_hazirla() -> pd.DataFrame:
    log.info("Gecmis CSV'ler okunuyor...")

    tuketim_df = pd.read_csv(TUKETIM_CSV, sep=";")
    uretim_df  = pd.read_csv(URETIM_CSV,  sep=";", skiprows=3)

    # Sutun adlarini temizle (bos bosluk, BOM karakteri vb.)
    tuketim_df.columns = tuketim_df.columns.str.strip()
    uretim_df.columns  = uretim_df.columns.str.strip()

    # Tuketim sutununu bul (encoding farkliligi olabilir)
    tuketim_kolon = next(
        (c for c in tuketim_df.columns if "ketim" in c and "MWh" in c),
        None
    )
    if tuketim_kolon is None:
        raise KeyError(f"Tuketim sutunu bulunamadi. Mevcut sutunlar: {list(tuketim_df.columns)}")

    tuketim_df = sayi_duzelt(tuketim_df, tuketim_kolon)
    for col in uretim_df.columns[2:]:
        uretim_df = sayi_duzelt(uretim_df, col)

    tuketim_df["zaman"] = pd.to_datetime(
        tuketim_df["Tarih"].astype(str) + " " + tuketim_df["Saat"].astype(str),
        dayfirst=True, errors="coerce"
    )
    uretim_df["zaman"] = pd.to_datetime(
        uretim_df["Tarih"].astype(str) + " " + uretim_df["Saat"].astype(str),
        dayfirst=True, errors="coerce"
    )

    uretim_kolonlar = ["zaman", "Toplam", "Gunes", "Ruzgar",
                       "Dogal Gaz", "Ithal Komur", "Linyit",
                       "Jeotermal", "Barajli", "Akarsu", "Biyokutle"]

    # Uretim CSV'sindeki gercek sutun adlarini esle
    kolon_eslesme = {}
    for hedef in uretim_kolonlar:
        if hedef == "zaman":
            continue
        for gercek in uretim_df.columns:
            if hedef.lower().replace(" ", "") in gercek.lower().replace(" ", "").replace("ü","u").replace("ş","s").replace("ğ","g").replace("ı","i").replace("ö","o").replace("ç","c"):
                kolon_eslesme[hedef] = gercek
                break

    # Zaman + eslesen kolonlari al
    mevcut_uretim = ["zaman"] + list(kolon_eslesme.values())
    mevcut_uretim = [c for c in mevcut_uretim if c in uretim_df.columns]

    df = pd.merge(
        tuketim_df[["zaman", tuketim_kolon]],
        uretim_df[mevcut_uretim],
        on="zaman",
        how="inner"
    )

    # Tuketim sutununu standart isme donustur
    if tuketim_kolon != "Tuketim Miktari(MWh)":
        df.rename(columns={tuketim_kolon: "Tuketim Miktari(MWh)"}, inplace=True)

    # Uretim sutunlarini da standartlastir
    ters_eslesme = {v: k for k, v in kolon_eslesme.items()}
    df.rename(columns=ters_eslesme, inplace=True)

    df.dropna(subset=["zaman"], inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values("zaman", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Turetilmis metrikler
    gunes_s  = df["Gunes"]  if "Gunes"  in df.columns else pd.Series(0, index=df.index)
    ruzgar_s = df["Ruzgar"] if "Ruzgar" in df.columns else pd.Series(0, index=df.index)
    dg_s     = df["Dogal Gaz"]   if "Dogal Gaz"   in df.columns else pd.Series(0, index=df.index)
    ik_s     = df["Ithal Komur"] if "Ithal Komur" in df.columns else pd.Series(0, index=df.index)
    lin_s    = df["Linyit"]      if "Linyit"      in df.columns else pd.Series(0, index=df.index)

    df["Temiz_Enerji"] = gunes_s + ruzgar_s
    df["Fosil_Enerji"] = dg_s + ik_s + lin_s
    df["Yesil_Oran"]   = np.where(
        df["Toplam"] > 0,
        (df["Temiz_Enerji"] / df["Toplam"]) * 100,
        0.0
    )
    df["Karbon_gCO2_kWh"] = np.where(
        df["Toplam"] > 0,
        (df["Fosil_Enerji"] / df["Toplam"]) * 550,
        0.0
    )

    log.info(f"{len(df)} saatlik veri hazirland. "
             f"Aralik: {df['zaman'].min()} -> {df['zaman'].max()}")
    return df


# --- ISTATISTIK OZETI ---
def istatistik_yazdir(df_canli: pd.DataFrame) -> None:
    if len(df_canli) < 2:
        return
    log.info(
        f"Istatistik | "
        f"Toplam yayin: {len(df_canli)} saat | "
        f"Ort. tuketim: {df_canli['Tuketim Miktari(MWh)'].mean():,.0f} MWh | "
        f"Ort. yesil: %{df_canli['Yesil_Oran'].mean():.1f}"
    )


# --- ANA AKIS ---
def main():
    log.info("Akilli Sebeke Canli Veri Botu Baslatildi")
    log.info(f"Mod: {HIZLANDIRMA_MODU.upper()} | Bekleme: {BEKLEME_SURESI}s/saat | Dongu: {DONGU_MU}")

    try:
        tum_veri = veri_hazirla()
    except Exception as e:
        log.critical(f"Veri yuklenemedi: {e}", exc_info=True)
        sys.exit(1)

    if os.path.exists(CANLI_DOSYA):
        os.remove(CANLI_DOSYA)
        log.info(f"Eski {CANLI_DOSYA} silindi.")

    ilk_veri = tum_veri.iloc[:ILK_PENCERE]
    ilk_veri.to_csv(CANLI_DOSYA, index=False)
    log.info(f"Ilk {ILK_PENCERE} saatlik taban veri panele yuklendi.")
    log.info("CANLI YAYIN BASLIYOR...")

    kalan_veri  = tum_veri.iloc[ILK_PENCERE:]
    yayin_sayisi = 0
    tur          = 1

    while _calis:
        for _, row in kalan_veri.iterrows():
            if not _calis:
                break

            try:
                anlik = pd.DataFrame([row])
                anlik.to_csv(CANLI_DOSYA, mode="a", header=False, index=False)
                yayin_sayisi += 1

                if yayin_sayisi % 10 == 0:
                    log.info(
                        f"[{row['zaman']}] Tur {tur} | "
                        f"{row['Tuketim Miktari(MWh)']:,.0f} MWh | "
                        f"%{row['Yesil_Oran']:.1f} yesil | "
                        f"{row.get('Ruzgar', 0):,.0f} MWh ruzgar"
                    )
                else:
                    print(
                        f"\r[{row['zaman']}]  {row['Tuketim Miktari(MWh)']:>9,.0f} MWh "
                        f"| %{row['Yesil_Oran']:>5.1f}  ",
                        end="", flush=True
                    )

            except Exception as e:
                log.warning(f"Satir yazilamadi: {e} - atlaniyor.")

            time.sleep(BEKLEME_SURESI)

        if not DONGU_MU or not _calis:
            break

        tur += 1
        zaman_kaydirma = tum_veri["zaman"].max() - tum_veri["zaman"].min()
        tum_veri = tum_veri.copy()
        tum_veri["zaman"] = tum_veri["zaman"] + zaman_kaydirma
        kalan_veri = tum_veri.iloc[ILK_PENCERE:]
        log.info(f"Tur {tur} basladi. Zaman kaydirildi: +{zaman_kaydirma}")

    print()
    if os.path.exists(CANLI_DOSYA):
        df_son = pd.read_csv(CANLI_DOSYA)
        istatistik_yazdir(df_son)
    log.info(f"Bot duzgun kapandi. Toplam yayin: {yayin_sayisi} saat.")


if __name__ == "__main__":
    main()
