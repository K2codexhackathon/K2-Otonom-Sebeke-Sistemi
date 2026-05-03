# Akıllı Şebeke Komuta Merkezi

Elektrik şebekelerini geleneksel yapısından kurtarıp, otonom ve akıllı bir sisteme çeviren dijital ikiz projesidir. Sistem, canlı veri işleme, makine öğrenmesi ile yük tahmini ve kriz anlarında yapay zeka destekli otonom kurtarma planları sunar.

## Çözülen Temel Sorunlar
* **Dinamik Yük Şokları (Veri Körlüğü):** Geleneksel şebekelerin anlık dalgalanmalara karşı kör kalması problemini canlı telemetri ile aşar.
* **Manuel Hantallık (Otonom Kriz Eksikliği):** Frekans çökmesi risklerine karşı insan müdahalesini beklemek yerine Gemini AI ile saniyeler içinde otonom aksiyon alır.

## Sistem Özellikleri
* **Canlı Telemetri:** `bot.py` üzerinden saniye saniye şebeke verisi simülasyonu.
* **Makine Öğrenmesi Tahmini:** Scikit-learn Gradient Boosting ile gelecek 24 saatin tüketim ve yenilenebilir enerji üretimi tahmini.
* **Senaryo Simülasyonu (What-If):** Beklenmedik hava olaylarının (sıcaklık, rüzgar, bulutluluk) şebekeye anlık etkisinin test edilmesi.
* **Yapay Zeka Destekli Otonom Aksiyon:** Kriz durumlarında Gemini AI entegrasyonu ile anında çözüm ve müdahale raporlaması.

## Kurulum Adımları

Projeyi sorunsuz bir şekilde bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayın.

### 1. Kütüphanelerin Kurulumu
Projenin ihtiyaç duyduğu tüm dış modülleri ve yapay zeka araçlarını tek seferde kurmak için terminali açın ve aşağıdaki komutu çalıştırın:
```bash
pip install pandas numpy streamlit scikit-learn plotly requests google-generativeai

https://drive.google.com/file/d/1u6-MUhtAvwP7Me3JWw4nEiuanzD0dCtY/view?usp=drive_link
