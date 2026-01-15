# Survival Double Descent: Stress-Testing Generalization in Time-to-Event Models

## 📌 Project Overview

This repository contains the code and experimental framework for **"The 'Survival Double Descent': Stress-Testing Deep Learning Generalization in Time-to-Event Analysis."**

While the "Double Descent" phenomenon—where test error decreases, spikes at the interpolation threshold (), and decreases again in the over-parameterized regime ()—is well-documented in classification and regression, its existence in **Survival Analysis** remains underexplored. This project uses rigorous synthetic data generation to determine if deep survival models (like DeepSurv) benefit from massive over-parameterization or if they suffer from catastrophic overfitting in high-noise clinical settings.

### Key Hypotheses

1. **Interpolation Threshold**: In survival analysis, the critical peak of overfitting is driven by the **number of events** (), not the total sample size ().
2. **Metric Divergence**: Rank-based metrics (**C-index**) often fail to detect overfitting that calibration metrics (**Brier Score**) reveal clearly.
3. **Data Sensitivity**: Skewed (log-normal) and high-cardinality features exacerbate the "critical regime" spike.

---



## 📊 Experimental Scenarios

This project implements four specific stress tests as defined in the research proposal:

| Scenario | Code | Description | Hypothesis |
| --- | --- | --- | --- |
| **A: Baseline** | `run_baseline.py` | . Standard setup. | Standard double descent curve (). |
| **B: Skewed** | `run_skew.py` | . Mimics biomarkers. | Peak variance increases; "benign overfitting" is harder to achieve. |
| **C: Sparse** | `run_sparse.py` | High-cardinality categorical features. | One-hot encoding causes early peaking; Embeddings flatten the curve. |
| **D: Imbalance** | `run_censor.py` | 90% Censoring rate. | The interpolation peak shifts left to . |

---

## 📈 Evaluation Metrics

We track two primary metrics to detect the divergence between **discrimination** and **calibration**:

1. **Concordance Index (C-index)**: Measures ranking ability. Often insensitive to the "variance spike" at the interpolation threshold.
2. **Integrated Brier Score (IBS)**: Measures the mean squared error of probability predictions. We expect this to show the catastrophic failure at the critical threshold ().

---

## 📚 References

If you use this code or methodology, please cite the following core references:

* **Survival Double Descent Theory**: *Liu, Y., Cai, J., & Li, D. (2025). Understanding Overparametrization in Survival Models through Double-Descent. arXiv preprint arXiv:2512.12463.*
* **Deep Double Descent**: *Nakkiran, P., et al. (2021). Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics.*
* **DeepSurv Architecture**: *Katzman, J. L., et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC medical research methodology.*
