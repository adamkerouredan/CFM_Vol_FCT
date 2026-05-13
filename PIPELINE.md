# PIPELINE COMPLET — CFM VOLATILITY FORECASTING CHALLENGE

> **Objectif** : prédire la volatilité réalisée d'actions américaines sur la fenêtre 14h–16h
> à partir des données intraday observées entre 9h30 et 13h55.
> **Métrique officielle** : MAPE
> **Référence méthodologique** : G. Paleologo, *The Elements of Quantitative Investing*, Wiley 2024.

---

## STATUT GLOBAL

| Phase | Statut | Description |
|-------|--------|-------------|
| Phase I   | ✅ TERMINÉ | Ingestion + EDA |
| Phase II  | ✅ TERMINÉ | Feature Engineering (10 features finales) |
| Phase III | ✅ TERMINÉ | Modélisation + Validation industrielle |
| Phase IV  | ✅ TERMINÉ | Soumission — Rang 8 / 17, MAPE = 24.25 |

---

## ARCHITECTURE DU PROJET

```
CFM_Vol_FCT/
├── data/
│   ├── X_train/training_input.csv      (gitignore)
│   ├── X_test/testing_input.csv        (gitignore)
│   └── Y_train.csv                     (gitignore)
├── src/
│   ├── data_loader.py                  ✅
│   ├── eda_analyzer.py                 ✅
│   ├── feature_engineer.py             ✅
│   ├── neutralizer.py                  ✅
│   ├── splitter.py                     ✅
│   ├── validator.py                    ✅
│   ├── evaluator.py                    ✅
│   └── model.py                        ✅
├── notebooks/
│   └── main.ipynb                      ✅
├── outputs/                            (gitignore)
├── PIPELINE.md
└── README.md
```

---

## PHASE I — INGESTION & EDA ✅

### Résultats clés
- **636 313 observations** sur 318 stocks × 2 117 jours
- **15.7 % de lignes** avec au moins 1 NaN
- **NaN concentrés à 09h30** (28 091 cas) → signal microstructure
- **TARGET** : skewness = 5.01, kurtosis = 59.9 → log-transform justifiée
- **Persistance volatilité** : ρ(vol_matin, TARGET) = **0.857**
- **Profil intraday en L** : 0.584 (09h30) → 0.154 (13h00)

### Décisions
- ✅ Imputation par interpolation linéaire intraday + ffill + bfill
- ✅ Log-transformation de la TARGET (Harvey-Shephard)
- ✅ Pas de winsorisation des features brutes (outliers économiques réels)
- ✅ Tri par `date` abandonné (jours randomisés par CFM)

---

## PHASE II — FEATURE ENGINEERING ✅

### Feature set final — 10 features

| Feature | IC (Pearson) | Kendall τ | IC cross-sect. | t-stat |
|---------|-------------:|----------:|---------------:|-------:|
| vol_mean              | +0.768 | +0.629 | +0.747 | +484 |
| vol_mean_recent       | +0.678 | +0.552 | +0.622 | +324 |
| vol_std               | +0.653 | +0.517 | +0.586 | +290 |
| vol_last_bar          | +0.534 | +0.368 | +0.391 | +191 |
| vol_mean_minus_median | +0.519 | +0.357 | +0.371 | +159 |
| vol_min               | +0.562 | +0.337 | +0.407 | +154 |
| vol_linear_slope      | -0.421 | -0.271 | -0.362 | -127 |
| return_n_negative     | +0.211 | +0.136 | +0.162 | +51  |
| return_n_positive     | +0.149 | +0.089 | +0.139 | +45  |
| vol_recent_over_mean  | +0.192 | +0.135 | +0.090 | +37  |

### Suppressions documentées
- **vol_median**            → r = 0.969 avec vol_mean (Pearson)
- **vol_trend**             → r = 0.918 avec vol_linear_slope (Pearson)
- **vol_range**             → r = 0.999 avec vol_max (Spearman)
- **vol_max**               → r = 0.956 avec vol_std (Spearman)
- **return_direction_bias** → IC = -0.048 (sans signal)
- **return_last_bar**       → IC = +0.002 (sans signal)

### Marchenko-Pastur — composantes réelles
- λ₊ = 1.0087
- **3 composantes au-dessus du bruit** : PC1 (6.37), PC2 (1.79), PC3 (1.20)

---

## PHASE III — MODÉLISATION ✅

### Résultats des modèles

| Modèle | MAPE (CV) | MAPE (holdout) |
|--------|----------:|---------------:|
| Baseline 1 (mean 54 bars) | 0.3713 | 0.3686 |
| Baseline 4 (1h mean) | - | 0.2921 |
| Ridge A (10 features) | 0.3090 | - |
| Ridge B (PCA 3 composantes) | 0.3365 | - |
| HAR-RV (Corsi 2009) | 0.2732 | - |
| LightGBM C (10 features) | 0.2600 | - |
| LightGBM D (108 raw bars) | 0.2584 | 0.2569 |
| LightGBM E (+ product_id) | 0.2574 | - |
| Stacking HAR + LightGBM | 0.2614 | - |
| **LightGBM F (30 structured)** | **0.2427** | **0.2392** |

### Champion — LightGBM F
- 30 features : 10 brutes + 10 Z-score par date (MAD) + 10 Z-score par stock
- Validation : Repeated Stratified K-Fold (5 folds × 2 répétitions)
- Holdout 15 % intouchable jusqu'à évaluation finale
- Bootstrap 1000× groupé par date
- QLIKE holdout : 0.1928
- R² : 0.78

### Limitations identifiées
- Underestimation systématique en régime haute volatilité (66% decile 10)
- Dégradation aux queues extrêmes (top 0.1% : MAPE = 29.7%)
- Biais MAPE sur les jours calmes (Q1 tire la moyenne vers le haut)

---

## PHASE IV — SOUMISSION ✅

- **Score public Challenge Data** : MAPE = 24.25
- **Rang** : 8 / 17
- **Benchmark officiel** : 68.14 (battu par facteur 2.8×)

---

## RÈGLES STRICTES

### Validation
- ❌ Jamais de fit du scaler/PCA sur le fold de validation
- ❌ Jamais d'early stopping sur le fold de test
- ❌ Pas de cherry-picking de modèles a posteriori
- ✅ Le holdout 15 % est intouchable jusqu'à la toute fin

### Reproducibilité
- ✅ `random_state` fixé partout
- ✅ Logging systématique de chaque expérience
- ✅ Versionnage Git de tous les commits significatifs

### Honnêteté méthodologique
- ✅ Toutes les pistes abandonnées sont documentées
- ✅ Les résultats négatifs sont reportés
- ✅ Les faiblesses du modèle ne sont pas masquées

---

## SOURCES

- **G. Paleologo**, *The Elements of Quantitative Investing*, Wiley 2024.
- **F. Corsi**, A Simple Approximate Long-Memory Model of Realized Volatility, JFEC 2009.
- **A. C. Harvey, N. Shephard**, Estimation of an Asymmetric Stochastic Volatility Model, JBES 1996.
- **T. G. Andersen, L. Benzoni**, Realized Volatility, Handbook of Financial Time Series, Springer 2009.
- **A. J. Patton, K. Sheppard**, Evaluating Volatility and Correlation Forecasts, Springer 2009.
- **V. Marchenko, L. Pastur**, Distribution of Eigenvalues for Some Sets of Random Matrices, 1967.
- **M. López de Prado**, *Advances in Financial Machine Learning*, Wiley 2018.

---

**Dépôt GitHub** : `adamkerouredan/CFM_Vol_FCT`
**Auteur** : Adam Kerouredan — M2 Quantitative Finance, 2025-2026.
