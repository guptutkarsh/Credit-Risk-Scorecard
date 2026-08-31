# Credit Risk PD Model & Scorecard — Lending Club

An end-to-end Probability of Default (PD) model, application scorecard, IFRS 9 staging,
and Expected Credit Loss (ECL) engine built on Lending Club loan data. Developed and
documented in line with **SR 11-7** model risk management principles — covering model
development, validation, assumptions, and limitations.

---

## 1. Executive Summary (Risk Committee View)

A logistic regression PD model was developed on **1.26 million** Lending Club accepted loans
to rank borrower default risk, scale it into a usable credit score, and quantify expected
loss under IFRS 9.

| Dimension | Result |
|---|---|
| Development sample | 1,260,344 loans · 882,240 train / 378,104 test · base default rate **19.20%** |
| Model type (champion) | Logistic Regression on Weight-of-Evidence features (7 predictors) |
| Discrimination | AUC **0.6958** · GINI **0.3916** · KS **28.2** |
| Calibration | Brier **0.1432** (beats 0.16 base-rate benchmark) |
| Stability | Vintage **PSI < 0.10** across all origination years 2010–2018 (population stable) |
| Sign integrity | **All 7 coefficients negative** as WoE construction requires (§4.3) |
| IFRS 9 stage default rates | Stage 1 **9.37%** → Stage 2 **21.46%** → Stage 3 **37.69%** |
| Portfolio ECL coverage | **9.25%** of exposure (elevated; drivers documented in §7) |

**Recommendation:** The model discriminates risk reliably and produces a monotonic,
defensible scorecard suitable for a cutoff-based lending policy. It is fit for a
**mid-tier decisioning** use case subject to the limitations in §7 (PD term structure,
LGD calibration, ongoing monitoring) being addressed before production.

---

## 2. Business Problem

Unsecured consumer lending requires a model that (a) ranks applicants by default risk,
(b) translates risk into an actionable score and approve/reject cutoff, and (c) quantifies
expected loss for provisioning under IFRS 9. This project delivers all three.

---

## 3. Data

- **Source:** Lending Club accepted loans (2007–2018), `accepted_2007_to_2018Q4.csv.gz`
- **Volume:** first 2,000,000 rows read (151 raw columns) → **1,260,344** loans after censoring
- **Target definition:** `loan_status` → 1 if *Charged Off*, *Default*, or
  *Does not meet the credit policy: Charged Off*; else 0. **Base default rate = 19.20%**
  (242,031 bad / 1,018,313 good)
- **Outcome censoring:** 739,656 `Current` loans removed — outcome not yet realised, including
  them would contaminate the target (standard development practice)
- **Candidate features:** 18 application-time attributes (loan, income, bureau, employment,
  home ownership, purpose, grade) — no post-origination or repayment fields, so no leakage
- **Missing values:** median imputation on `emp_length` (6.0), `revol_util` (52.8), `dti` (17.62),
  `mort_acc` (1.0), `pub_rec_bankruptcies` (0.0); median over mean because financial
  distributions are right-skewed. Largest gaps were `emp_length` 5.79% and `mort_acc` 3.97%
- **Split:** 70% train / 30% test, stratified on target, fixed seed 42 (reproducible)

---

## 4. Methodology

### 4.1 Feature Engineering — Weight of Evidence (WoE)
- All features transformed to WoE: `WoE = ln(%good / %bad)` per bin
- Numeric features binned into 10 quantile bins; categoricals binned per level
- Bin edges opened at ±∞ so unseen test values always land in a bin
- **Smoothing:** +0.5 Laplace correction on good/bad counts to avoid `log(0)` in sparse bins
- WoE/IV implemented **from scratch** (pandas), not via library — full transparency over
  binning, smoothing, and leakage control
- **Leakage control:** all bins learned on train only, then applied unchanged to test
  (sklearn-style fit/transform separation)

**Information Value ranking (train), threshold IV < 0.02 → drop:**

| Feature | IV | | Feature | IV |
|---|---|---|---|---|
| grade | **0.4517** | | purpose | 0.0195 ✂ |
| int_rate | **0.4334** | | inq_last_6mths | 0.0191 ✂ |
| dti | 0.0705 | | emp_length | 0.0054 ✂ |
| loan_amnt | 0.0313 | | open_acc | 0.0052 ✂ |
| home_ownership | 0.0293 | | revol_bal | 0.0036 ✂ |
| annual_inc | 0.0293 | | delinq_2yrs | 0.0015 ✂ |
| mort_acc | 0.0282 | | pub_rec | 0.0012 ✂ |
| installment | 0.0265 | | total_acc | 0.0009 ✂ |
| revol_util | 0.0257 | | pub_rec_bankruptcies | 0.0006 ✂ |

Nine features dropped on IV, leaving 9 — reduced to **7** after the diagnostics in §4.3.
`grade` and `int_rate` carry almost all the predictive power (IV > 0.43 each; the rest are
"weak" by the conventional 0.02–0.10 band).

### 4.2 Champion vs Challenger

| Model | Test AUC | Decision |
|---|---|---|
| **Champion — Logistic Regression on WoE** | **0.6958** | Selected for production |
| Challenger — Gradient Boosting (sklearn defaults) | 0.7007 | Rejected; retained as benchmark |

The challenger wins by **+0.0049 AUC**. That gain was judged not to offset the validation and
governance cost of a black-box model: the champion gives coefficient-level explainability,
monotonic WoE inputs, and satisfies SR 11-7 documentation standards. The challenger is retained
as an ongoing performance benchmark — if the gap widens materially at the next revalidation,
the champion is the one under review.

### 4.3 Model Diagnostics (documented validation loop)

The first fit flagged two problems, both resolved before the model was accepted:

- **`installment` dropped** — coefficient −0.016, **p = 0.627**. It is arithmetically derived from
  `loan_amnt`, `int_rate` and `term`, all already represented, so it carries no independent
  signal. Textbook multicollinearity.
- **`revol_util` dropped** — its univariate WoE is clean and monotonic (bad rate rises 14.1% →
  22.2% across bins) and its **VIF is only 1.12**, so this was *not* collinearity. In the
  multivariate fit its coefficient turned **positive (+0.081)** — a **suppressor effect**: its
  conditional contribution given `dti`/`int_rate`/`grade` is genuinely, slightly reversed.
  The effect is tiny, but a positive coefficient would award scorecard points in the wrong
  direction, so it was removed for **sign integrity**. (At 882k rows, p < 0.05 is cheap —
  significance was not treated as evidence the sign was trustworthy.)
- **`int_rate` ↔ `grade` redundancy** — the real collinearity in the model, surfaced only by VIF:
  **9.60 / 9.49** (on Lending Club, grade determines the interest rate). Both sit under the
  VIF > 10 "severe" threshold, both are significant and correctly signed, so **both retained**
  with the redundancy documented rather than silently dropped.

**Final champion — 7 features, all coefficients negative (higher WoE = safer = lower PD):**

| Term | Coefficient | | Term | Coefficient |
|---|---|---|---|---|
| const | −1.4372 | | dti | −0.4972 |
| loan_amnt | −1.0144 | | mort_acc | −0.4625 |
| grade | −0.6876 | | int_rate | −0.2326 |
| annual_inc | −0.6432 | | | |
| home_ownership | −0.6316 | | | |

Pseudo R² = 0.079, LLR p-value = 0.000, converged in 6 iterations.
(`outputs/champion_coefficients.csv`)

### 4.4 Scorecard Scaling
- Standard scaling: **Score = Offset + Factor × ln(odds)**, odds = (1 − PD) / PD
- Parameters: PDO = 20, base score = 600 at 50:1 odds → **Factor = 28.85, Offset = 487.12**
- Realised score distribution on test: **min 473 · median 532 · max 604** (sd 22.4)
- Monotonicity confirmed — actual default rate falls cleanly across all ten score deciles:

| Score band | 473–505 | 505–514 | 514–520 | 520–526 | 526–532 | 532–538 | 538–545 | 545–553 | 553–567 | 567–604 |
|---|---|---|---|---|---|---|---|---|---|---|
| Actual default rate | 41.7% | 30.9% | 25.8% | 22.2% | 18.9% | 15.9% | 13.2% | 11.1% | 7.7% | **4.8%** |

An 8.8× spread between the worst and best decile — the scorecard separates risk in the
direction and order a lending policy needs.

### 4.5 IFRS 9 Staging & ECL
- **Three-stage model:** Stage 1 (12-month ECL) → Stage 2 SICR (lifetime ECL) →
  Stage 3 impaired (lifetime ECL)
- SICR proxied by PD thresholds (PD < 0.15 → Stage 1; 0.15–0.30 → Stage 2; ≥ 0.30 → Stage 3)
  — simplification noted in §7
- **ECL = PD × LGD × EAD** | LGD = 0.45 (Basel foundation IRB, senior unsecured) | EAD = loan amount
- Stage default rates confirm clean risk separation, and the provisioning table concentrates
  loss in Stage 3 exactly as expected:

| Stage | Loans | Exposure (USD) | ECL (USD) | Coverage | Actual default rate |
|---|---|---|---|---|---|
| 1 | 157,351 | 2.046 bn | 85.1 m | 4.2% | **9.37%** |
| 2 | 156,079 | 2.285 bn | 225.0 m | 9.8% | **21.46%** |
| 3 | 64,662 | 1.159 bn | 197.7 m | 17.1% | **37.69%** |
| **Total** | **378,092** | **5.489 bn** | **507.8 m** | **9.25%** | 19.20% |

Stage 3 is 17% of loans but 39% of ECL.

---

## 5. Validation Results

| Metric | Value | Bucket | Interpretation |
|---|---|---|---|
| AUC | 0.6958 | Discrimination | Ranks risk well above random |
| GINI | 0.3916 | Discrimination | Industry-convention rescaling of AUC |
| KS | 28.2 | Discrimination | Max good/bad separation; informs cutoff |
| Brier | 0.1432 | Calibration | Beats 0.16 base-rate benchmark — PDs are informative |
| PSI (train-test) | 0.0000 | Stability | Same-period random split — confirms no split bias only |
| PSI (vintage) | **< 0.10** all years | Stability | Score distribution stable across 2010–2018 origination cohorts |

Validation spans the three model-risk dimensions: **discrimination, calibration, stability.**
(`outputs/validation_metrics.csv`)

**KS / gains detail.** Peak separation of 28.2 occurs in the 4th-worst decile. The worst three
score bands hold 30% of the book but capture **51% of all defaulters** — the lift statement a
credit policy is actually built on. (`outputs/ks_gains_table.csv`)

**Out-of-time stability (vintage PSI).** The test-set predictions were re-bucketed by loan
origination year and PSI computed against a 2010 base cohort (years with n ≥ 2,000 only).
Every vintage from 2010 to 2018 scored **PSI < 0.10** (peak 0.087 in 2013) — i.e. the model's
score distribution does not drift across eight years of originations, so no PSI-based
recalibration would be triggered. This is the *out-of-time* test that a train-test PSI cannot
provide, and it is why the train-test PSI of 0.0000 is reported but not relied on.

| Vintage | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|
| PSI vs 2010 | 0.040 | 0.030 | 0.087 | 0.075 | 0.044 | 0.046 | 0.070 | 0.062 |

---

## 6. Business Decision Layer

A cutoff-sensitivity analysis links the score to lending policy: as the approval cutoff rises,
approval rate falls while the approved-book default rate and ECL both decline — quantifying the
volume-vs-loss tradeoff management dials in by risk appetite.

| Score cutoff | Approval rate | Default rate of approved book | ECL of approved book (USD) |
|---|---|---|---|
| 473 (approve all) | 100.0% | 19.20% | 507.8 m |
| 503 | 91.8% | 17.05% | 394.9 m |
| 513 | 81.0% | 15.10% | 294.4 m |
| 523 | 65.5% | 12.70% | 191.5 m |
| **533** | **48.3%** | **10.29%** | **110.0 m** |
| 543 | 32.4% | 8.17% | 57.1 m |
| 553 | 20.3% | 6.25% | 27.7 m |
| 573 | 5.2% | 4.01% | 4.4 m |

Reading the table: a cutoff of 533 halves the book but cuts expected loss by **78%** and the
approved-book default rate from 19.2% to 10.3%. Where the line actually goes is a risk-appetite
decision, not a modelling one — the model's job is to price the tradeoff.

---

## 7. Assumptions & Limitations (Model Risk Disclosure)

1. **PD term structure** — a single PD is applied across stages; Stage 1 should use a
   12-month PD. The target is a through-the-life default flag (closer to lifetime PD),
   which overstates Stage 1 ECL. Production requires separate 12-month and lifetime PD curves.
2. **LGD** — fixed at 0.45 (regulatory assumption); production should estimate LGD from
   recovery/charge-off data.
3. **EAD** — proxied by loan amount; acceptable for term loans, but a true EAD would account
   for amortisation and (for revolving products) credit conversion factors.
4. **SICR definition** — proxied by PD thresholds; true IFRS 9 compares reporting-date PD
   to origination PD.
5. **Out-of-time validation** — addressed via vintage PSI (all origination years < 0.10, §5).
   Production still requires *ongoing* PSI monitoring on live scoring data with a formal
   rebuild trigger (e.g. PSI > 0.25) and seasonality controls.
6. **ECL coverage (9.25%)** — elevated versus a typical 3–8% consumer book, driven by
   removal of `Current` loans, Lending Club's risk profile, and the conservative LGD.
7. **Missing-value bin in `grade`** — 15 loans have a null grade and form their own WoE bin.
   With zero observed defaults, Laplace smoothing assigns them WoE **+1.997**, i.e. the
   *safest* bin in the model. The population is negligible (0.002%) so results are unaffected,
   but a production build must floor or merge sparse bins rather than let a 15-loan cell earn
   the best score in the scorecard.
8. **Residual missingness** — 501 rows retain nulls in core fields after median imputation
   (~0.04%). WoE maps these to a neutral 0.0 rather than dropping them.
9. **Sample truncation** — the first 2,000,000 rows of the source file are read rather than the
   full extract. The retained window still spans 2007–2018 originations and 1.26m outcomes,
   but it is not the complete population.
10. **Model power is grade-dominated** — `grade` and `int_rate` supply IV > 0.43 each while every
    other retained feature is under 0.032. The model is close to a re-expression of Lending
    Club's own underwriting decision, which caps achievable AUC and would not transfer to a
    lender without an equivalent pre-existing grade.
11. **Reject inference not applied** — the sample contains accepted loans only, so the model is
    fitted on a through-the-door population already filtered by Lending Club's credit policy.

---

## 8. Repository Structure

```
credit-risk-scorecard/
├── data/                     # raw + derived datasets (not committed — see .gitignore)
├── notebooks/
│   └── Project 1.ipynb       # full build: data prep → WoE/IV → model → validation → ECL
├── outputs/
│   ├── champion_coefficients.csv   # final 7-feature model
│   ├── validation_metrics.csv      # AUC, GINI, KS, Brier, PSI
│   ├── ks_gains_table.csv          # decile gains / KS table
│   └── *.pkl                       # model + WoE bins (not committed — retain training data)
└── README.md
```

## 9. Reproducing

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy joblib
```

Download `accepted_2007_to_2018Q4.csv.gz` from the Lending Club Kaggle dataset into `data/`,
then run `notebooks/Project 1.ipynb` top to bottom. Fixed random seed (42) ensures
reproducibility. Note that file paths in the notebook are absolute Windows paths and need
repointing on another machine.

---

*Author: Utkarsh Gupta — FRM Part I & II Passed (GARP, 2025–2026), MBA (IIM Kozhikode).
Built as an applied credit-risk modelling exercise with full SR 11-7 documentation.*
