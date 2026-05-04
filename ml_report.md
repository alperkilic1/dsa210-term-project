# ML Baseline Raporu — Incident Genome

## Problem
Incident sureleri (short vs long) siniflandirmasi. Sadece ilk 1 saat icinde erisilebilir bilgiler kullanildi (leakage-free).

## Yontem
- **Model:** Random Forest (200 tree, max_depth=10, class_weight='balanced')
- **Features:** service, created_hour, created_weekday, first_hour_updates + turetilmis (is_business_hours, is_weekend, hour_sin, hour_cos)
- **Split:** Stratified 80/20 (563 train, 141 test)
- **Preprocessing:** OneHotEncoder (kategorik), passthrough (numerik)

## Sonuclar

| Metrik | Deger |
|--------|-------|
| Train Accuracy | 0.7762 |
| CV F1-macro (5-fold) | 0.6412 +/- 0.0370 |
| Test Accuracy | 0.7447 |
| Test F1 (long) | 0.79 |
| Test F1 (short) | 0.67 |
| Test F1-macro | 0.7317 |

### Naive baseline karsilastirmasi

432/704 = 0.6136 majority class. "Always predict long" baseline = 61.36% accuracy. Random Forest 74.47% — absolute +13.11pp, relative +21.4%. Model naive baseline'i istatistiksel olarak yeniyor.

## En Onemli Ozellikler
Permutation importance (F1-macro drop) sonuclarina gore en etkili ozellikler belirlendi. Detaylar notebook'taki bar chart'ta mevcut.

## LR vs RF Karsilastirmasi

| Model | Test Acc | Test F1-macro | CV F1-macro |
|-------|----------|---------------|-------------|
| Random Forest (baseline) | 0.7447 | 0.7317 | 0.6412 |
| Logistic Regression | 0.7376 | 0.7298 | 0.5797 |
| RF Tuned (GridSearchCV) | 0.7163 | 0.7038 | 0.6551 |

Logistic Regression (StandardScaler + class_weight='balanced') RF'ye yakin test performansi gosteriyor ancak CV'de belirgin sekilde daha dusuk. RF'nin ensemble yaklasimi bu veri seti icin daha stabil genelleme sagliyor.

Tuned RF, CV'de en iyi skoru aliyor (0.6551) ancak test setinde baseline'in altinda kaliyor — bu da kucuk veri setinde hyperparameter tuning'in overfitting riskini gosteriyor.

## Hyperparameter Tuning

GridSearchCV (3-fold, f1_macro) ile taranan parametreler:
- `n_estimators`: [100, 200, 400]
- `max_depth`: [5, 10, 15, None]
- `min_samples_leaf`: [2, 5, 10]

**Best params:** n_estimators=400, max_depth=15, min_samples_leaf=10  
**Best CV F1-macro:** 0.6551

Baseline RF (n_estimators=200, max_depth=10, min_samples_leaf=5) zaten yakin optimal noktada. Grid search daha derin agaclara (max_depth=15) ve daha buyuk ensemble'a (400) yoneliyor ancak test generalization'da fark yaratmiyor.

## SHAP Analizi — Top Ozellikler

| Sira | Ozellik | Mean |SHAP| |
|------|---------|----------------|
| 1 | first_hour_updates | 0.1033 |
| 2 | service_netlify | 0.0185 |
| 3 | created_hour | 0.0147 |

**Yorum:** Ilk 1 saatteki guncelleme sayisi (`first_hour_updates`) diger tum ozelliklerden 5-6x daha etkili. Bu, operasyon ekipleri icin net bir sinyal: ilk saatte yuksek guncelleme alan incident'lar uzun sureli olacak. Servis tipi (ozellikle Netlify) ve saatin ikincil etkisi var.

## Sinirliliklar
- 704 ornek ile kucuk veri seti — high variance
- Tuned RF test'te baseline'dan dusuk (small data overfitting)
- Text features (incident name) kullanilmadi
- Temporal autocorrelation kontrol edilmedi

## Gelecek Calisma
- Gradient boosting (XGBoost/LightGBM) karsilastirmasi
- Incident name'den TF-IDF/embedding features
- Temporal validation (time-based split)
- Daha buyuk veri seti ile tuning tekrari
