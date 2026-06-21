# Credit Risk PD Model & Scorecard — Lending Club

An end-to-end Probability of Default (PD) model, application scorecard, IFRS 9 staging,
and Expected Credit Loss (ECL) engine built on Lending Club loan data. Developed and
documented in line with **SR 11-7** model risk management principles — covering model
development, validation, assumptions, and limitations.

---

## 1. Executive Summary (Risk Committee View)

A logistic regression PD model was developed on Lending Club's accepted-loans portfolio
to rank borrower default risk, scale it into a usable credit score, and quantify expected
loss under IFRS 9.

| Dimension | Result |
|---|---|
| Model type (champion) | Logistic Regression on Weight-of-Evidence features |
| Discrimination | AUC **0.696** · GINI **0.392** · KS **28.2** |
| Calibration | Brier **0.143** (beats 0.16 base-rate benchmark) |
| Stability | PSI ≈ 0 (in-sample; out-of-time validation noted as production requirement) |
| IFRS 9 stage default rates | Stage 1 **9.4%** → Stage 2 **21.5%** → Stage 3 **37.7%** |
| Portfolio ECL coverage | **9.25%** of exposure (elevated; drivers documented in §7) |

**Recommendation:** The model discriminates risk reliably and produces a monotonic,
defensible scorecard suitable for a cutoff-based lending policy. It is fit for a
**mid-tier decisioning** use case subject to the limitations in §7 (out-of-time
validation, PD term structure, LGD calibration) being addressed before production.

---

## 2. Business Problem

Unsecured consumer lending requires a model that (a) ranks applicants by default risk,
(b) translates risk into an actionable score and approve/reject cutoff, and (c) quantifies
expected loss for provisioning under IFRS 9. This project delivers all three.

---

## 3. Data

- **Source:** Lending Club accepted loans (2007–2018), `accepted_2007_to_2018Q4.csv.gz`
- **Target definition:** `loan_status` → 1 if *Charged Off / Default*, else 0
- **Outcome censoring:** `Current` loans removed — outcome not yet realised, including them
  would contaminate the target (standard development practice)
- **Split:** 70% train / 30% test, stratified on target, fixed seed (reproducible)

---

## 4. Methodology

### 4.1 Feature Engineering — Weight of Evidence (WoE)
- All features transformed to WoE: `WoE = ln(%good / %bad)` per bin
- Numeric features binned into quantiles; categoricals binned per level
- **Information Value (IV)** used for feature selection (threshold: drop IV < 0.02)
- WoE/IV implemented **from scratch** (pandas), not via library — full transparency over
  binning, smoothing (+0.5 Laplace correction to avoid log(0)), and leakage control
- **Leakage control:** all bins learned on train only, then applied unchanged to test
  (sklearn-style fit/transform separation)

### 4.2 Champion vs Challenger
- **Champion:** Logistic Regression — selected for full interpretability, coefficient-level
  explainability, and SR 11-7 documentation compliance
- **Challenger:** Gradient Boosting — retained as a performance benchmark; rejected for
  production despite marginally higher AUC because its black-box nature raises validation
  and governance cost that the marginal gain does not justify

### 4.3 Model Diagnostics (documented validation loop)
- **`installment` dropped** — insignificant (p = 0.63), collinear with loan_amnt × int_rate × term
- **`revol_util` dropped** — univariate WoE monotonic and correct, VIF low (1.12, not collinear),
  but exhibited a conditional sign reversal (suppressor effect) producing a counterintuitive
  positive coefficient; removed to preserve correct risk direction in the scorecard
- **`int_rate` ↔ `grade` redundancy** — VIF ≈ 9.6 / 9.5 (grade drives interest rate on
  Lending Club); both retained (VIF < 10) with redundancy documented

### 4.4 Scorecard Scaling
- Standard scaling: **Score = Offset + Factor × ln(odds)**
- Parameters: PDO = 20, base score = 600 at 50:1 odds → **Factor = 28.85, Offset = 487.12**
- Score validated as monotonic against actual default rate across deciles

### 4.5 IFRS 9 Staging & ECL
- **Three-stage model:** Stage 1 (12-month ECL) → Stage 2 SICR (lifetime ECL) →
  Stage 3 impaired (lifetime ECL)
- SICR proxied by PD thresholds (0.15 / 0.30) — simplification noted in §7
- **ECL = PD × LGD × EAD** | LGD = 0.45 (Basel foundation IRB, senior unsecured) | EAD = loan amount
- Stage default rates confirm clean risk separation: **9.4% → 21.5% → 37.7%**

---

## 5. Validation Results

| Metric | Value | Bucket | Interpretation |
|---|---|---|---|
| AUC | 0.696 | Discrimination | Ranks risk well above random |
| GINI | 0.392 | Discrimination | Industry-convention rescaling of AUC |
| KS | 28.2 | Discrimination | Max good/bad separation; informs cutoff |
| Brier | 0.143 | Calibration | Beats 0.16 base-rate benchmark — PDs are informative |
| PSI | ~0.00 | Stability | In-sample only; out-of-time is the real test |

Validation spans the three model-risk dimensions: **discrimination, calibration, stability.**

---

## 6. Business Decision Layer

A cutoff-sensitivity analysis links the score to lending policy: as the approval cutoff
rises, approval rate falls while the approved-book default rate and ECL both decline —
quantifying the volume-vs-loss tradeoff management dials in by risk appetite.
(See `outputs/cutoff_sensitivity.csv`.)

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
5. **Out-of-time validation** — PSI computed in-sample only; a later-cohort (out-of-time)
   PSI is required to demonstrate true stability and is the production monitoring trigger
   (rebuild at PSI > 0.25).
6. **ECL coverage (9.25%)** — elevated versus a typical 3–8% consumer book, driven by
   removal of `Current` loans, Lending Club's risk profile, and the conservative LGD.

---

## 8. Repository Structure

```
credit-risk-scorecard/
├── data/            # raw + processed datasets (not committed — see .gitignore)
├── notebooks/       # Sessions 1–6 development notebooks
├── outputs/         # model, validation metrics, scored portfolio, cutoff table
└── README.md
```

## 9. Reproducing

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy joblib
```
Run the notebooks in order (Sessions 1–6). Fixed random seed (42) ensures reproducibility.

---

*Author: Utkarsh Gupta — FRM Part I & II Passed (GARP, 2025–2026), MBA (IIM Kozhikode).
Built as an applied credit-risk modelling exercise with full SR 11-7 documentation.*
