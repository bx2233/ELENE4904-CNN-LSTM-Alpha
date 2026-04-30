# CNN-LSTM Alpha Model on S&P 500

**ELENE 4904 · Statistical Learning in Quantitative Trading · Spring 2026**

This repository contains the full code and evaluation pipeline for our final project: a CNN-LSTM-Attention sequence model that produces a cross-sectional alpha signal on S&P 500 stocks, evaluated against four standard baselines under a realistic long-short backtest with transaction costs and risk constraints.

---

## Team

| Name | UNI | Role |
|------|-----|------|
| Beibei Xian | bx2233 | Data acquisition, feature engineering, signal-quality diagnostics, presentation |
| Carlos Kuchenmeister | cmk2250 | Model architecture (CNN-LSTM-Attention), training pipeline, baselines, presentation |
| Neill Gonzales | nag2177 | Backtesting engine, portfolio construction, performance metrics, attribution |
| Anurag Chatterjee | ac5929 | Project framing, agenda structuring, write-up review |
| Liang Song | ts3479 | Data scoping, dataset sanity checks, split-logic review |

---

## Headline results (out-of-sample, Jan 2023 – Apr 2026)

| Metric | CNN-LSTM-Attn | Ridge | XGBoost | Momentum | Mean Reversion |
|---|---|---|---|---|---|
| **Mean Rank IC** | **0.0362** | 0.0109 | 0.0223 | -0.0004 | 0.0006 |
| **IC t-stat** | **5.46** | 1.66 | 4.00 | -0.06 | 0.10 |
| Positive IC % | 56.99% | 51.24% | 54.12% | 51.64% | 47.39% |

**Portfolio (CNN-LSTM-Attn long/short)**

| Metric | Value |
|---|---|
| Total Return | +40.02% |
| Annualized Return | 11.73% |
| Annualized Volatility | 19.33% |
| Sharpe Ratio | 0.67 |
| Sortino Ratio | 1.17 |
| Calmar Ratio | 0.63 |
| Max Drawdown | -18.66% |
| Recovery Period | 248 days |
| Annualized Alpha vs SPY | 13.20% |
| Net Beta | 0.5190 |
| # Trades | 3,478 |
| Win Rate | 54.89% |
| Avg Holding (days) | 29.8 |

---

## How to run

### 1. Clone and install

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Open the notebook

The full pipeline lives in a single notebook:

```
notebooks/code+eval.ipynb
```

Open it in Google Colab or Jupyter and **Run All**. End-to-end runtime is roughly 25–40 minutes on a Colab T4 GPU (most of the time is the Yahoo Finance download and the LSTM training).

### 3. Outputs

All evaluation plots are saved automatically to `./eval_plots/` and the metrics tables print inline at the end of the notebook.

---

## Pipeline overview

The notebook is organized into the following sections:

**1. Imports & config** — sets random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) and global hyperparameters:

```python
START_DATE = "2014-01-01"
LOOKBACK = 60
HORIZON = 5
TRADE_LAG = 1
TRAIN_END = "2020-12-31"
VAL_END = "2022-12-31"
LONG_Q = 0.90
SHORT_Q = 0.10
REBALANCE_EVERY = 5
TRANSACTION_COST = 0.001
MAX_POSITION = 0.08
```

**2. Data acquisition** — pulls the current S&P 500 constituent list from Wikipedia, joins GICS sectors, and downloads daily OHLCV for every ticker from Yahoo Finance via `yfinance`. SPY is downloaded separately as the benchmark for excess-return and beta features. Tickers with fewer than 1,000 trading days of history are dropped.

**3. Feature engineering** — 24 features in 6 groups, defined in `add_features()`:

| Group | Features |
|---|---|
| Returns | `ret_1d / 5d / 10d / 21d / 63d`, `excess_ret_1d / 5d` vs SPY, `ret_5d / 21d / 63d` cross-rank |
| Volatility | `vol_5d / 21d / 63d`, `vol_21d` cross-z + percentile, `beta_63` vs SPY |
| Price vs SMA | `price_to_sma_10 / 21 / 63`, cross-sectional z-score, `daily_range`, `close_position` |
| Volume / Liquidity | `volume_chg_1d`, `volume_to_sma_21`, `dollar_volume_to_sma_21` |
| Microstructure | `close_open_ret`, `daily_range`, cross-stock dispersion |
| Cross-section | per-date z-score and percentile rank applied to 8 base features |

The prediction target is the **5-day forward return with a 1-day execution lag** — a signal formed on day T is traded on T+1.

**4. Sequence construction** — `make_sequences()` builds (lookback, num_features) tensors per stock-day. Train: Jan 2014 – Dec 2020 (≈773k sequences). Validation: 2021–2022 (≈218k). Test: Jan 2023 – Apr 2026 (≈379k).

**5. Model — CNN-LSTM-Attention** — defined in `build_advanced_model()`:

```
Input (60, 24)
  → Conv1D(32, kernel=3, padding=same, ReLU)
  → LayerNormalization
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → Attention: Dense(1, tanh) → softmax → weighted sum
  → Dense(32, Swish)
  → Dropout(0.1)
  → Dense(1)  # predicted alpha
```

Trained with Adam (lr=0.0005), Huber loss, batch size 512, up to 30 epochs with EarlyStopping (patience 5, monitor `val_loss`, `restore_best_weights=True`).

**6. Baselines** — fitted on the last timestep of each sequence (tabular):

- `Ridge(alpha=10.0)`
- `XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)` — falls back to `HistGradientBoostingRegressor` if XGBoost is not available
- 21-day momentum (raw `ret_21d`)
- 5-day mean reversion (negated `ret_5d`)

**7. Signal quality** — `rank_ic_series()` and `ic_summary()` compute the daily Spearman rank correlation between predicted scores and realized 5-day returns, plus the IC t-statistic (Newey-West-style standard error not needed at this volume of observations).

**8. Backtesting engine** — `build_portfolio_history()`:
- Long the top decile (90th percentile) of predicted scores, short the bottom decile (10th).
- Inverse-volatility weighting using `vol_21d`, then `capped_normalize()` enforces an 8% per-stock cap and renormalizes the long and short books separately.
- Rebalances every 5 trading days; charges 0.1% one-way on all weight changes.
- Tracks per-stock contributions, gross/net exposure, beta, and turnover at every rebalance.

**9. Evaluation suite** — `performance_summary()`, `period_metrics()`, `trade_metrics()`, `risk_metrics()`, and `yearly_performance()` compute the full set of metrics shown in the tables above. All plots are written to `./eval_plots/`:

---

## Methodology notes

- **Survivorship bias** — we use the *current* S&P 500 list as our universe and require ≥1000 trading days of history per ticker. This is a known limitation; a fully survivorship-bias-free study would also include constituents that were added and later removed during the window. We discuss this in the Future Work slide of the deck.
- **Lookahead bias** — the `TRADE_LAG = 1` parameter ensures that a signal formed at the close of day T is only traded on day T+1, simulating realistic execution.
- **Train-only normalization** — `StandardScaler` is fit only on the training set so future statistics cannot leak in. Cross-sectional z-scoring is applied per date, which is leakage-safe by construction.
- **Reproducibility** — seeds are set for both NumPy and TensorFlow. Note that some non-determinism remains from cuDNN kernels and async data shuffling — re-runs may produce results that differ by a few basis points. Set `tf.config.experimental.enable_op_determinism()` for fully deterministic runs at the cost of speed.

---

## Repository layout

```
.
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── EE4904code+eval.ipynb                # main pipeline (data → model → backtest → metrics)
│   └──  EE4904code+eval.ipynb-Colab.pdf/                        # notebook with out put and 
├── EE4904_Presentation.pptx # final deck
├── EE4904_Presentation pptx.pdf  # PDF export of the deck
├── lstm_preds_sp500.csv
└── .gitignore
```

---

## Dependencies

```
numpy
pandas
scipy
yfinance
requests
matplotlib
scikit-learn
xgboost            # optional; falls back to HistGradientBoostingRegressor
tensorflow >= 2.10
```

Install with `pip install -r requirements.txt`.

---

## Acknowledgements

This project was developed for ELENE 4904 (Statistical Learning in Quantitative Trading) at Columbia University, Spring 2026. We thank the course staff for guidance on backtesting methodology and for feedback on the signal-quality diagnostics.
