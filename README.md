

##  Project: DRW - Crypto Market Prediction (Kaggle Competition)

### **Overview**

This project tackles the Kaggle competition hosted by DRW, which challenges participants to predict short-term crypto market movements using a mix of anonymized proprietary features and public market data. The goal is to generate a directional signal for future price changes—specifically, to predict the `label` column provided in the dataset.

### **Objective**

Build a robust predictive model capable of extracting structure from noisy, high-dimensional time series data. The model should predict the direction of crypto price movement based on historical market signals, using both statistical and machine learning techniques.

---

##  Exploratory Data Analysis (EDA)

### **Data Quality**

* The dataset is clean: no missing values across the 896 features.
* The time series is well-sampled and rich, covering nearly 1 year of minute-level data.

### **Target Analysis**

* The target `label` resembles the behavior of Bitcoin returns: noisy, heavy-tailed, and volatile.
* Stationarity confirmed by the Augmented Dickey-Fuller (ADF) test.
* ACF tails off and PACF cuts off → Initial assumption: AR(p) process is plausible.

### **Correlation & Mutual Information**

* Pearson correlation between `label` and other features is weak.

  * This suggests a low linear signal-to-noise ratio.
  * Justifies using tree-based models (e.g. XGBoost, LightGBM) that can model non-linear interactions.
* Mutual Information analysis highlights a subset of features with useful predictive signal (e.g. `X853`, `X854`, `X862`).

### **Outlier & Distributional Properties**

* Multiple outlier detection methods (IQR, Z-Score, MAD) show \~4–7% extreme values.
* Label distribution exhibits long tails and high kurtosis, reinforcing the need for robust modeling techniques.

### **Dimensionality Reduction**

* PCA analysis shows that \~95% of variance can be captured with a reduced set of components.
* Motivates dimensionality reduction to combat the curse of dimensionality and improve training time.

### **Time Series Modeling**

* AR(p) models tested with Ljung-Box test reveal residual autocorrelation.
* This motivates moving from pure AR to ARIMA(p, d, q) models to better capture temporal dependencies.

---

##  Modeling & Performance

### **Models Used**

* **XGBoost**: Tuned with walk-forward validation and feature selection via MI. Performs well on non-linear signals.
* **LightGBM**: Faster and more scalable; slightly less robust to extreme outliers than XGBoost.

### **Validation Strategy**

* Walk-forward (rolling origin) used to avoid data leakage.
* Evaluation metric: Pearson correlation (competition metric).
* Public leaderboard score: **\~0.110**

### **Key Engineering Choices**

* Feature selection via Mutual Information thresholding.
* Robust scaling of features.
* Extensive feature analysis and redundancy removal (e.g. correlation filtering).
* Experiments with rolling statistics and lagged features for time awareness.

---

##  Next Steps

* Explore hybrid ARIMA + Gradient Boosting ensemble.
* Try sequence models (e.g. Temporal Convolutional Networks or Transformers) with lagged features.
* Examine feature drift over time to optimize window-based training.

---

