# 📈 Stock Price Forecasting — LSTM Time-Series Model

**Sequence-based LSTM neural network for financial time-series prediction**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-LSTM-red.svg)](https://keras.io/)

---

## 📌 Overview

This project applies **Long Short-Term Memory (LSTM)** networks to predict Google stock prices using historical market data. LSTMs capture long-range sequential dependencies — making them well-suited for non-stationary financial time series.

University of Adelaide · Deep Learning Assessment 3 · **Grade: 87 / 100**

---

## 📊 Dataset — Google Stock Price (Kaggle)

| File | Use |
|---|---|
| `Google_Stock_Price_Train.csv` | Model training & validation |
| `Google_Stock_Price_Test.csv` | Final evaluation (unseen data) |

- **Feature used:** `Open` price only (normalised with MinMaxScaler to [0, 1])
- **Sequence construction:** sliding window — input: `seq_length` consecutive prices → target: next price
- **Train / Validation split:** 80% / 20%

---

## 🏗️ Model Architecture

```
Input  → (batch_size, seq_length=15, features=1)
LSTM   → 50 hidden units (returns sequences)
LSTM   → 50 hidden units
Dense  → 1 output (predicted next price)
```

| Parameter | Value |
|---|---|
| LSTM layers | 2 (stacked) |
| Hidden units per layer | 50 |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 0.01 (best found) |

---

## 📈 Results

### Regression Metrics (seq_length = 15)

| Metric | Value |
|---|---|
| MSE | 382.96 |
| RMSE | 19.57 |
| MAE | 18.56 |

### Sequence Length Comparison

| seq_length | MSE | RMSE | MAE |
|---|---|---|---|
| **5** | **130.78** | **11.44** | **7.65** |
| 10 | 256.14 | 16.00 | 12.36 |
| 15 | 382.96 | 19.57 | 18.56 |

> Shorter sequences capture short-term trends more effectively for this dataset.

### Key Findings
- Training loss steadily decreased over 50 epochs — model converged well
- Validation loss tracks training loss with no significant divergence — good generalisation
- Residuals skewed positive — model slightly under-predicts during sharp upward moves
- Best hyperparameters: lr = 0.01, epochs = 50

---

## 🔍 Visualisations Produced

1. **Actual vs Predicted prices** — shows trend capture quality
2. **Training & Validation loss curves** — confirms convergence
3. **Residual histogram** — distribution of prediction errors
4. **Moving average overlay** — trend smoothing analysis

---

## 🚀 How to Run

```bash
git clone https://github.com/chandr-csgp/Assign3.git
cd Assign3

pip install tensorflow numpy pandas matplotlib scikit-learn

python main.py
```

---

## 🔧 Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `pandas` · `Matplotlib` · `scikit-learn`

---

## 👤 Author

**Chandra Sekar Putta** — University of Adelaide, MDS 2024–2026  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
