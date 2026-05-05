# ML Baseline Raporu — Incident Genome

**Code:** [`ml_baseline.ipynb`](ml_baseline.ipynb) · **Raw metrics:** [`data/ml_results.json`](data/ml_results.json) · **AI assistance:** see [README.md § AI assistance](README.md#ai-assistance).

## Arastirma Sorusu (RQ)

> Sadece bir incident'in ilk 1 saati icinde gozlemlenebilir ozelliklerle (servis, baslangic saati, gun, ilk-saat guncelleme sayisi) outage'in **kisa (< 60 dk) mi yoksa uzun (>= 60 dk) mi** olacagini tahmin edebilir miyiz?

## Problem
Incident sureleri (short vs long) ikili siniflandirma. Sadece ilk 1 saat icinde erisilebilir bilgiler kullanildi (leakage-free).

## Yontem
- **Models karsilastirilan:** Random Forest baseline, Logistic Regression (GridSearchCV), Random Forest (GridSearchCV-tuned)
- **Baseline RF:** 200 tree, max_depth=10, class_weight='balanced'
- **Features:** `service`, `created_hour`, `created_weekday`, `first_hour_updates` + turetilmis (`is_business_hours`, `is_weekend`, `hour_sin`, `hour_cos`)
- **Split:** Stratified 80/20 (563 train, 141 test), `random_state=42`
- **Preprocessing:** OneHotEncoder (kategorik), passthrough/StandardScaler (numerik)
- **CV:** 5-fold StratifiedKFold (model-karsilastirma), 3-fold GridSearchCV (tuning)
- **Scoring:** F1-macro (her iki sinifa esit agirlik)

### CV strategy notu

StratifiedKFold tercih edildi — TimeSeriesSplit yerine — cunku (a) incident'lar bagimsiz servis kesintileridir, (b) cogu servis sadece son 12-24 ay veri sergiliyor (kucuk temporal pencere), (c) sinif dengesini her fold'da koruma onceligi. Temporal validation gelecek calisma olarak listelendi (asagi).

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

| Metrik | Always-predict-long (DummyClassifier) | RF baseline | Fark |
|--------|--------------------------------------:|------------:|-----:|
| Accuracy | 0.6170 | 0.7447 | +0.1277 (+20.7% relative) |
| F1-macro | 0.3816 | 0.7317 | +0.3501 |

Always-predict-long F1-macro 0.38'de takiliyor cunku short sinifinda recall = 0 (F1 = 0). RF her iki sinifi da tahmin edebildigi icin F1-macro farki accuracy farkindan cok daha buyuk (+0.35 vs +0.13).

**Anlamlilik:** 141 ornekli test setinde RF'nin 105 dogru tahmini, training prior `p=0.6128` H0 altinda one-sided binomial test ile **p = 6.67e-04** (notebook cell 19'da `scipy.stats.binomtest(105, 141, 0.6128, alternative='greater')` ile hesaplandi). Effect size **Cohen's h = 0.284** (small-to-medium pratik iyilestirme).

Confusion matrix: `figures/ml_confusion_matrix.png` (Blues colormap, dpi=200, bold annotations).

## En Onemli Ozellikler

Permutation importance (F1-macro drop) sonuclarina gore en etkili ozellikler `first_hour_updates`, `service_*` (en bilgilendiriciler Netlify ve OpenAI), `created_hour`. Detay: `figures/ml_feature_importance.png`.

![Feature importance — top permutation contributors](figures/ml_feature_importance.png)

## LR vs RF Karsilastirmasi

| Model | Test Acc | Test F1-macro | CV F1-macro |
|-------|----------|---------------|-------------|
| Random Forest (baseline) | 0.7447 | 0.7317 | 0.6412 +/- 0.037 |
| Logistic Regression (GridSearchCV over C) | 0.7092 | 0.6973 | 0.6033 +/- 0.060 |
| RF Tuned (GridSearchCV) | 0.7163 | 0.7038 | 0.6551 (best) |

Logistic Regression (StandardScaler + class_weight='balanced' + GridSearchCV `C in [0.01, 0.1, 1, 10, 100]`, lbfgs/L2) en iyi `C=0.01` (en yuksek regularization) ile tunelandi — lineer modelin zayif sinyali baski altinda kalabilmesi icin. Test F1-macro 0.6973 RF baseline 0.7317'ye yakin ama CV'de belirgin sekilde dusuk (0.6033 vs 0.6412), RF'nin ensemble yaklasiminin bu veri seti icin daha stabil genelleme sagladigini gosteriyor.

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

| Sira | Ozellik | Mean \|SHAP\| |
|------|---------|----------------|
| 1 | first_hour_updates | 0.1033 |
| 2 | service_netlify | 0.0185 |
| 3 | created_hour | 0.0147 |

(Binary RF icin |SHAP| iki sinifta da ayni; isaret farkliligi hangi sinifa pozitif kontribu ettigini gosterir.)

![SHAP summary — first_hour_updates dominates](figures/ml_shap_summary.png)

**Yorum:** Ilk 1 saatteki guncelleme sayisi (`first_hour_updates`) diger tum ozelliklerden 5-6x daha etkili. Bu, operasyon ekipleri icin net bir sinyal: ilk saatte yuksek guncelleme alan incident'lar uzun sureli olacak. Servis tipi (ozellikle Netlify) ve saatin ikincil etkisi var.

## Tartisma (RQ'ya Cevap)

RQ "ilk 1 saat ozellikleriyle short/long ayrimi yapilabilir mi?" sorusuna cevap **evet, fakat sinirli**. RF F1-macro 0.7317 vs naive 0.3803 (p < 0.001) ile naive baseline'i kesin yenmesi sinyalin gercek oldugunu gosteriyor. Ancak CV F1-macro (0.64) ile test F1-macro (0.73) arasindaki ~0.09 fark kucuk veri setinin (704 ornek) yarattigi yuksek varyansi vurguluyor — production-grade degil, indikatif. `first_hour_updates`'in dominansi (5-6x) operasyonel deger sunuyor: ilk saatte yuksek guncelleme = uzun outage sinyali, triage onceliklendirmesinde kullanilabilir.

## Sinirliliklar
- 704 ornek ile kucuk veri seti — high variance
- Tuned RF test'te baseline'dan dusuk (small data overfitting)
- Text features (incident name) kullanilmadi
- Temporal autocorrelation kontrol edilmedi (StratifiedKFold rastgele karistiriyor)
- Confidence interval (bootstrap) test F1 uzerinde hesaplanmadi — sadece CV std mevcut

## Gelecek Calisma
- Gradient boosting (XGBoost/LightGBM) karsilastirmasi
- Incident name'den TF-IDF/embedding features
- Temporal validation (TimeSeriesSplit veya tarihsel cutoff)
- Daha buyuk veri seti ile tuning tekrari
- Bootstrap CI test metrikleri uzerinde

## Reproducibility

- **Python:** 3.10+ (gelistirme: 3.14)
- **Random seed:** `random_state=42` (train_test_split, RF, LR, StratifiedKFold, permutation_importance, GridSearchCV)
- **Kurulum:**
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Calistirma:**
  ```bash
  jupyter nbconvert --to notebook --execute ml_baseline.ipynb --output ml_baseline.ipynb
  ```
- **Beklenen ciktilar:** `data/ml_results.json` (metrikler), `figures/ml_confusion_matrix.png`, `figures/ml_feature_importance.png`, `figures/ml_shap_summary.png`
- **Veri girisi:** `data/incidents_clean.csv` (EDA notebook'undan uretilir)
