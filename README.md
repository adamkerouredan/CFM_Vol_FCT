# CFM Volatility Forecasting Challenge

> Predicting US equity afternoon realized volatility (2pm-4pm window) from
> morning intraday bars (9:30am-1:55pm), with rigorous quantitative research
> methodology grounded in Paleologo's *The Elements of Quantitative Investing*
> (Wiley, 2024).

---

## Final result

```
═══════════════════════════════════════════════════════════════════════
LightGBM F — Champion model

  MAPE on held-out test set     : 0.2392
  95% bootstrap CI (by date)    : [0.2352, 0.2451]
  QLIKE (Patton-Sheppard 2009)  : 0.1928
  R²                            : 0.80
  Improvement vs CFM baseline   : −34.9%
═══════════════════════════════════════════════════════════════════════
```

The champion model is a LightGBM regressor trained on 30 features structured
across three economic dimensions: raw signal, cross-sectional rank by date,
and historical rank by stock. The methodology follows Paleologo's Loadings
Generation framework (Chap. 6).

---

## Table of contents

1. [Problem context](#1-problem-context)
2. [Approach in five steps](#2-approach-in-five-steps)
3. [Champion model — why this one](#3-champion-model--why-this-one)
4. [Validation protocol](#4-validation-protocol)
5. [Known limitations](#5-known-limitations)
6. [Paths not taken — and why](#6-paths-not-taken--and-why)
7. [Repository structure](#7-repository-structure)
8. [How to reproduce](#8-how-to-reproduce)
9. [Bibliography](#9-bibliography)

---

## 1. Problem context

**Why predict volatility?** Future asset volatility is a foundational input
for portfolio construction (risk parity), risk management (Value-at-Risk),
position sizing (Kelly criterion), and option pricing (implied vs realized
spread). Capital Fund Management (CFM), a systematic hedge fund, organized
this challenge through the Inria/ENS *Projets Informatiques MASH* course.

**What's the task?** Given 54 intraday volatility bars (5-min frequency,
9:30am to 1:55pm) and 54 return-direction bars for 318 US stocks across
2,117 anonymized dates, predict the realized volatility of the 2pm-4pm
window. The official metric is Mean Absolute Percentage Error (MAPE).

**What makes this hard?**

- **Volatility is highly persistent.** A naive baseline (the morning mean)
  already explains 73% of the afternoon variance. The challenge is in the
  remaining 27%, which is genuinely difficult to capture out-of-sample.
- **MAPE is asymmetric.** When the true volatility is small (TARGET = 0.05),
  a small absolute error of 0.02 translates to 40% MAPE. The metric is
  dominated by quiet days, not stress days.
- **Dates are anonymized.** Inter-day chronological order is unavailable.
  Classical time-series approaches (walk-forward validation, GARCH, HAR
  inter-day) are not directly applicable.

---

## 2. Approach in five steps

The pipeline follows a disciplined progression from raw data to validated
champion model.

### Step 1 — Exploratory analysis and decisions

Key findings that drove subsequent choices:

- 15.7% of rows contain at least one NaN, concentrated at 9:30am
  (microstructure of the opening auction). NaNs were imputed via intraday
  linear interpolation, never with cross-row information.
- TARGET distribution is highly skewed (skewness 5.01, kurtosis 59.9)
  with minimum 0.000132. **Log-transformation was applied** following
  Harvey-Shephard (1996).
- Persistence: ρ(morning vol mean, afternoon vol) = 0.857. This sets the
  bar for any model — it must beat this benchmark by a meaningful margin.

### Step 2 — Feature engineering

10 hand-crafted features were constructed from the 54 volatility bars and
54 return-direction bars, covering five economic dimensions:

| Dimension | Features | Economic hypothesis |
|-----------|----------|---------------------|
| Global level | `vol_mean` | Volatility clustering (Mandelbrot 1963) |
| Asymmetry | `vol_mean_minus_median` | Presence of isolated spikes |
| Recency | `vol_mean_recent`, `vol_last_bar` | Proximity to target window |
| Trend | `vol_linear_slope` | Direction of intraday profile |
| Dispersion | `vol_std`, `vol_min` | Regime stability |
| Acceleration | `vol_recent_over_mean` | Recent vs global dynamic |
| Directional | `return_n_positive`, `return_n_negative` | Intraday directional bias |

Six initially constructed features were dropped after multicollinearity
audit (Spearman correlation > 0.9 with retained features) or insufficient
Information Coefficient (IC < 0.05).

### Step 3 — Feature evaluation

Three orthogonal metrics computed to validate the feature set:

- **Information Coefficient** (Pearson, Kendall τ, cross-sectional Spearman):
  all t-statistics > 35 on 2,117 dates → strong statistical robustness.
- **Marchenko-Pastur eigenvalue analysis**: identified exactly 3 components
  above the noise threshold (PC1=6.37, PC2=1.79, PC3=1.20 vs λ₊=1.0087).
- **Cross-sectional neutralization**: IC drops by ~13% on average when the
  market-wide component is removed, confirming that the signal is mostly
  idiosyncratic, not a beta proxy.

### Step 4 — Modeling (eight variants tested)

| Model | MAPE (CV) | Verdict |
|-------|----------:|---------|
| Baseline 1 (mean of 54 bars) | 0.3713 | CFM official baseline |
| Ridge A (10 features, winsorized) | 0.3090 | Linear reference |
| Ridge B (PCA on 3 components) | 0.3365 | Lost 24% variance — rejected |
| HAR-RV (Corsi 2009, 3 horizons) | 0.2732 | Academic benchmark |
| LightGBM C (10 features) | 0.2600 | Non-linearity adds value |
| LightGBM D (108 raw bars) | 0.2584 | Marginal gain over C |
| LightGBM E (+ product_id) | 0.2574 | LightGBM already learns stock implicitly |
| Stacking HAR-RV + LightGBM | 0.2614 | Errors correlated — rejected |
| **LightGBM F (30 structured features)** | **0.2427** | **Champion** |

### Step 5 — Industrial validation

The champion model was subjected to a complete validation protocol on a
15% held-out set never seen during training, model selection, or feature
engineering:

- Hold-out evaluation with bootstrap **grouped by date** (1000 iterations)
  to respect intra-day autocorrelation.
- Decile-by-decile underestimation analysis (critical for risk applications).
- Pairwise statistical comparison between models (test of dominance).
- QLIKE loss evaluation (Patton-Sheppard 2009): rank-robust alternative to
  MAPE for volatility forecasts.
- Top-tail audit (deciles 10%, 5%, 1%, 0.5%, 0.1%) to verify queue behavior.

---

## 3. Champion model — why this one

LightGBM F uses **30 features** structured across three economic levels,
each answering a different question:

```
Level 1 (10 features)  : "What is the absolute value of this feature?"
                          → raw engineered features

Level 2 (10 features)  : "Is this stock more volatile than the market today?"
                          → cross-sectional Z-score by date
                          → robust to outliers via median + MAD

Level 3 (10 features)  : "Is this stock more volatile than its own history?"
                          → historical Z-score by stock
                          → statistics fit on train only (no leakage)
```

This structure is directly inspired by **Paleologo (2024), Chap. 6 — Loadings
Generation**, which advocates expressing each raw signal in cross-sectional
and historical relative terms.

### Why this beats the alternatives

- **Versus Ridge** (best linear: 0.3090): non-linearities exist in the
  feature-target relationship (especially in `vol_mean`). LightGBM captures
  them; Ridge cannot.
- **Versus LightGBM C** (10 features only: 0.2600): the structured features
  explicitly encode information that LightGBM had to infer implicitly.
- **Versus LightGBM D** (108 raw bars: 0.2584): the 30 structured features
  carry the same information in a more compact, interpretable form, and
  beat the raw bars by 6.9%.
- **Versus stacking with HAR-RV** (0.2614): both models use the same
  underlying information (morning volatility), so their errors are
  correlated and ensembling provides no gain.

### Performance by regime

| Quartile | TARGET range | MAPE | Comment |
|----------|--------------|------|---------|
| Q1 (calm) | < 0.10 | 0.3573 | Worst — MAPE artifact on small values |
| Q2 | 0.10 - 0.14 | 0.2017 | Strong |
| Q3 | 0.14 - 0.21 | 0.1897 | Best |
| Q4 (volatile) | > 0.21 | 0.2083 | Strong — what matters for risk |
| **Top 10%** | > 0.33 | **0.2195** | **Better than global average** |

The model performs **better in high-volatility regimes than on average**,
which is the desired property for a risk-management application — accuracy
matters most precisely when the market is moving.

---

## 4. Validation protocol

### Cross-validation
**Repeated Stratified K-Fold** (5 folds × 2 repetitions = 10 evaluations),
stratified by TARGET quartile to ensure balanced regimes across folds.

### Hold-out
**15% of the dataset** stratified by TARGET quartile, set aside before any
modeling decision. Never touched until final evaluation.

### Bootstrap by date
**1,000 bootstrap iterations**, sampling dates (not rows) with replacement.
This respects the intra-date autocorrelation of observations. The resulting
confidence interval [0.2352, 0.2451] is tighter than the naive row-by-row
bootstrap, indicating high regime stability.

### Anti-leakage discipline
- Statistics by stock (Level 3 Z-scores) computed only on training folds.
- Cross-sectional statistics (Level 2 Z-scores) use only morning data, never
  TARGET — therefore safe to compute on the full dataset.
- Internal validation split (15% of training fold) for LightGBM early
  stopping, isolated from the external test fold.

### Sanity checks
- CV MAPE (0.2427) vs hold-out MAPE (0.2392): difference of −0.0015, well
  within the bootstrap CI. **No overfitting.**
- QLIKE classification consistent with MAPE classification: LightGBM F
  ranks first under both metrics, confirming robustness of the model
  selection.

---

## 5. Known limitations

The model has been audited honestly. Three limitations should be acknowledged
in any operational use:

### 5.1 Systematic underestimation in high-volatility regimes

```
Decile of TARGET     % observations underestimated
D1 (quietest)            6.9%
D5                      41.9%
D10 (most volatile)     66.4%
```

On the most volatile days, the model underestimates 2 out of 3 times. This
is the classical **shrinkage effect** of ML models trained with MSE loss
in log-space — predictions are pulled toward the mean. **In production
risk management, this bias would need correction** via multiplicative
recalibration per decile or asymmetric loss functions.

### 5.2 Degraded performance on the extreme tail

```
Top 1% of TARGET    MAPE = 24.4%
Top 0.5%            MAPE = 26.6%
Top 0.1%            MAPE = 29.7%
```

Performance degrades on true extreme events, though the absolute level
remains acceptable for the bulk of cases. An Extreme Value Theory (EVT)
overlay could improve this behavior.

### 5.3 MAPE-induced bias toward quiet days

Because the metric is relative to TARGET, the global MAPE of 24% is
mechanically dominated by quiet days (Q1 contributes disproportionately).
Q2-Q4 perform at ~20% but average is dragged up by Q1's 36%. This is a
**property of the metric, not of the model**.

---

## 6. Paths not taken — and why

Several promising directions were considered but excluded by deliberate
trade-off decisions. They are documented for completeness.

### 6.1 Sequential models (LSTM, TCN, Transformer)

**Expected gain**: 1-2 MAPE points by exploiting the temporal structure of
the 54 bars (which LightGBM treats as independent features).

**Excluded because**: significant loss of interpretability, weeks of
development time, and the project explicitly prioritized methodological
clarity over leaderboard ranking.

### 6.2 Classical time-series models (GARCH, EGARCH, HAR inter-day)

**Expected gain**: 0.5-1 point with proper inter-day chronological order.

**Excluded because**: the challenge anonymizes dates, making
inter-day temporal structure unavailable. Would require an external
dataset to apply.

### 6.3 MAPE-aware loss functions (sample_weight = 1/y²)

**Expected gain**: 2 points of global MAPE.

**Tested but rejected**: while global MAPE improved, the Q4 regime
performance collapsed (MAPE went from 0.21 to 0.30). For a risk-aware
model this trade-off is unacceptable. **The project prioritizes Q4
performance over leaderboard ranking.**

### 6.4 External data (VIX, market indices, alternative data)

**Expected gain**: 1-2 points.

**Excluded because**: only data provided by the challenge was used, by
methodological choice. Adding external data would require careful
data-acquisition discipline outside the scope of this project.

### 6.5 Diversified ensembling

**Expected gain**: 0.5-1.5 points if base models have decorrelated errors.

**Excluded because**: our stacking experiment (HAR-RV + LightGBM) showed
that models using similar input information have correlated errors. A
successful ensemble would require structurally different models (e.g.,
LightGBM + LSTM + GARCH), which falls under 6.1-6.2.

---

## 7. Repository structure

```
CFM_Vol_FCT/
├── data/                          # CSV files (gitignored)
├── src/
│   ├── data_loader.py             # DataLoader class
│   ├── eda_analyzer.py            # EDAAnalyzer class
│   ├── feature_engineer.py        # FeatureEngineer + FeatureTransformer
│   ├── neutralizer.py             # Neutralizer (date/stock)
│   ├── splitter.py                # Stratified train/holdout splitter
│   ├── evaluator.py               # MAPE + Jensen + diagnostics
│   ├── validator.py               # Repeated Stratified K-Fold
│   └── model.py                   # Ridge/Ridge+PCA/LightGBM/HAR/Stacking
├── notebooks/
│   └── main.ipynb                 # Complete pipeline notebook
├── outputs/                       # Generated figures (gitignored)
├── reports/                       # LaTeX report
├── PIPELINE.md                    # Project roadmap
├── FICHE_PASSATION.md             # Handover document (in French)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. How to reproduce

### Requirements

```
python      >= 3.11
numpy       >= 1.20
pandas      >= 2.0
scikit-learn >= 1.3
lightgbm    >= 4.0
matplotlib  >= 3.7
seaborn     >= 0.12
scipy       >= 1.10
```

### Running the pipeline

1. Place data files: `data/training_input.csv`, `data/Y_train.csv`,
   `data/testing_input.csv`.
2. `pip install -r requirements.txt`
3. Open `notebooks/main.ipynb` and run all cells sequentially.

The notebook is structured by phases (I to IX) with explicit markdown
sections, so partial reading is possible — each phase is self-contained
in its narrative.

---

## 9. Bibliography

### Primary methodological reference
- **G. Paleologo** (2024). *The Elements of Quantitative Investing*. Wiley.
  - Chap. 3 — Linear models of returns (Ridge, regularization)
  - Chap. 4 — Backtesting protocol (cross-validation, hold-out)
  - Chap. 5 — Forecast evaluation (QLIKE, Patton-Sheppard)
  - Chap. 6 — Loadings generation (Z-scores, winsorization)
  - Chap. 7 — Statistical factor models (PCA, Marchenko-Pastur)
  - Chap. 8 — Information coefficient

### Volatility forecasting
- **F. Corsi** (2009). *A Simple Approximate Long-Memory Model of Realized
  Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
- **A. C. Harvey, N. Shephard** (1996). *Estimation of an Asymmetric
  Stochastic Volatility Model for Asset Returns*. Journal of Business &
  Economic Statistics, 14(4), 429-434.
- **T. G. Andersen, L. Benzoni** (2009). *Realized Volatility*. Handbook
  of Financial Time Series, Springer.
- **A. J. Patton, K. Sheppard** (2009). *Evaluating Volatility and
  Correlation Forecasts*. Handbook of Financial Time Series, Springer.

### Foundational
- **B. Mandelbrot** (1963). *The Variation of Certain Speculative Prices*.
  Journal of Business, 36(4), 394-419.
- **R. F. Engle** (1982). *Autoregressive Conditional Heteroscedasticity
  with Estimates of the Variance of United Kingdom Inflation*.
  Econometrica, 50(4), 987-1007.
- **V. Marchenko, L. Pastur** (1967). *Distribution of Eigenvalues for
  Some Sets of Random Matrices*. Mathematics of the USSR-Sbornik, 1(4),
  457-483.

### Machine learning in finance
- **M. López de Prado** (2018). *Advances in Financial Machine Learning*.
  Wiley.

---

## Author

**Adam Kerouredan** — M2 Quantitative Finance, 2025-2026.

Project completed in approximately 5 days of intensive work, May 2026.
The code, methodology, and documentation are intended as a portfolio
demonstration of quantitative research discipline applied to a real-world
forecasting problem.

For questions or feedback, feel free to open an issue.
