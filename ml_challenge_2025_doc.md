# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Squads\
**Team Members:** Sachu Meghana, Vontela Gopika, Pulimamidi Shravani, Sreeperambudur Harini\
**Submission Date:** 13 October 2025

---

## 1. Executive Summary

This solution predicts optimal product prices using a lightweight, explainable LightGBM regression model trained on structured features extracted from textual catalog data. The approach focuses on parsing key price-influencing signals such as brand, pack quantity, weight, and quality terms. Despite being minimal and fast, it achieves a validation SMAPE of 65.93% and provides a strong foundation for multimodal extensions.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The dataset combined product titles, descriptions, and quantities, with price as the target. Exploratory Data Analysis (EDA) revealed that textual cues—like “Pack of N”, “organic”, and brand names—were highly correlated with prices. The data displayed large price variance and textual redundancy, suggesting a need for structured feature extraction.

**Key Observations:**

- Pack count and weight terms (e.g., “500g”, “Pack of 3”) strongly influenced pricing.
- Brand frequency acted as a proxy for market reputation.
- Descriptive terms (“premium”, “organic”) mapped to higher price brackets.

### 2.2 Solution Strategy

**Approach Type:** Single-Model Text-based Regression\
**Core Innovation:** Regex-based feature extraction and frequency-based brand encoding, followed by LightGBM regression on engineered features.

---

## 3. Model Architecture

### 3.1 Architecture Overview

**Workflow:**

1. Load train/test data.
2. Parse `catalog_content` for brand, IPQ (Item Pack Quantity), weight, and keywords.
3. Encode brand frequency and derive `ipq_weight = ipq × weight`.
4. Log-transform prices to stabilize skewed distribution.
5. Train LightGBM regressor → predict → exponentiate back to price.

### 3.2 Model Components

**Text Processing Pipeline:**

- **Preprocessing:** Regex extraction for pack count & weight, detection of “organic”/“premium” terms, brand frequency encoding.
- **Feature Engineering:** `brand_freq`, `ipq`, `weight`, `is_organic`, `is_premium`, `ipq_weight`.
- **Model Type:** LightGBM (GBDT).
- **Key Parameters:** `num_leaves=31`, `learning_rate=0.05`, `n_estimators=100`, `objective='regression'`, `metric='mae'`.

**Image Processing Pipeline:**

- Not used in this version; reserved for future multimodal integration.

---

## 4. Model Performance

### 4.1 Validation Results

- **SMAPE Score:** 65.93% (validation set)
- **MAE:** 0.6787 (log-scale)
- Early stopping did not trigger; model converged at 100 iterations.

---

## 5. Conclusion

The model efficiently captures product-price relationships from text features with minimal preprocessing and achieves a validation SMAPE of 65.93%. Its interpretability, reproducibility, and speed make it a strong baseline for further improvement—such as integrating TF-IDF embeddings or image features for finer granularity.

---

## Appendix

### A. Code Artefacts

[Access the complete code and model artifacts here](https://drive.google.com/drive/folders/1-kKzQN9Are6NJvmAsh1-Qbt8MCCLmvG2?usp=drive_link)

