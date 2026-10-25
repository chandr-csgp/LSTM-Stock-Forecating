# Google Stock Price Forecasting — LSTM (PyTorch)

A two-layer LSTM that predicts the next day's Google **opening price**
from the previous 5 days' opening prices. Written in PyTorch for a University
of Adelaide deep learning course.

All code is in a single notebook cell: [`RNN.ipynb`](RNN.ipynb).

---

## Data

The notebook expects two CSVs from the Kaggle *Google Stock Price* dataset in
the same folder (they are **not** committed to this repo):

| File | Rows | Used for |
|---|---|---|
| `Google_Stock_Price_Train.csv` | 2012–2016 daily prices | training + validation |
| `Google_Stock_Price_Test.csv` | January 2017 | test |

Only the `Open` column is used.

## Method

1. **Scaling** — `MinMaxScaler` to [0, 1].
2. **Sequences** — sliding window of `seq_length = 5` days → the next day's price.
   Train and test sequences are built separately, so the first 5 test days
   only serve as inputs.
3. **Split** — training sequences are split 80/20 into train/validation with
   `train_test_split(random_state=42)`.
4. **Model** — `LSTMStockPredictor`:

   ```
   input (batch, 5, 1)
   → nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
   → last time step → nn.Linear(50, 1)
   ```

5. **Training** — MSE loss, Adam (lr 0.01), 20 epochs, batch size 64.
6. **Evaluation** — predictions are inverse-scaled back to dollars and scored
   with MSE, RMSE and MAE on the test set.

The notebook plots actual vs predicted test prices, the train/validation loss
curves, and a histogram of test residuals.

## How to run

```bash
git clone https://github.com/chandr-csgp/LSTM-Stock-Forecating.git
cd LSTM-Stock-Forecating
pip install torch numpy pandas scikit-learn matplotlib jupyter
# put Google_Stock_Price_Train.csv and Google_Stock_Price_Test.csv here
jupyter notebook RNN.ipynb
```

Runs in seconds on a CPU.

## Results

The notebook was committed without outputs, so no metrics are recorded in this
repo — run it to reproduce them.

## Known limitations

These are worth knowing before reading anything into the error numbers:

- **Scaler fitted on train + test.** `MinMaxScaler` is fitted on all prices,
  including the test period, so test information leaks into preprocessing.
  It should be fitted on the training data only.
- **Shuffled validation split.** `train_test_split` shuffles sequences, so
  validation windows are interleaved with training windows in time and
  overlap them. A chronological split (last 20% of the training period) would
  give an honest validation score.
- **Very small test set.** The January 2017 file has about 20 trading days,
  leaving roughly 15 test predictions.
- **No naive baseline.** Next-day prices are close to today's price, so a
  "tomorrow = today" forecast is a strong baseline that the LSTM should be
  compared against.
- **Single feature.** Only `Open` is used — no volume, high/low or close.

## Tech

Python · PyTorch · pandas · NumPy · scikit-learn · Matplotlib

---

**Chandra Sekar Putta** — Master of Data Science, University of Adelaide ·
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
