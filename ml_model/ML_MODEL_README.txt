================================================================================
ML_MODEL — LSTM Oil Direction Pipeline
================================================================================

This document describes the LSTM-based pipeline that predicts short-term US oil
price direction from sequences of scraped news articles.  It is written for two
audiences: readers new to machine learning (plain-language explanations) and
readers who want implementation and design detail.


================================================================================
1. WHAT THIS PROJECT DOES
================================================================================

The system answers a focused question: after reading oil-related news from
several consecutive calendar days, will the oil market go up, stay roughly flat,
or go down on the next trading day we care about?

Inputs are not prices during the window — they are the *text* of news articles
saved as HTML under raw_articles/.  Each article is converted to a numerical
embedding (a list of 768 numbers from FinBERT).  For each day in the window we
average embeddings from up to ten articles.  Those daily vectors form a short
sequence fed into a bidirectional LSTM with attention.

Outputs are three classes:

  Class 0 — Down   : the configured oil proxy (default USO ETF) fell by more
                     than the flat band (default 0.5%) versus the prior close.
  Class 1 — Flat   : the move was inside the band (noise / no clear direction).
  Class 2 — Up     : the proxy rose by more than the flat band.

Why news matters for short-term direction: headlines and body text reflect
supply shocks (OPEC, sanctions), demand signals (China, refining), geopolitical
risk, and inventory commentary before prices fully adjust.  The model does not
“understand” markets like a trader; it learns statistical patterns between
language in the window and the next day’s realised return bucket.

Limitations (read these before trusting outputs):

  • Small sample count relative to model capacity — hundreds of windows, not
    millions.  Metrics can look good on one split and fail on new dates.
  • Labels come from a single ETF (USO), not every crude contract (WTI, Brent).
  • Articles are scraped unevenly by day; empty days become zero vectors.
  • FinBERT is frozen; only the LSTM head is trained — no domain adaptation of
    the text encoder.
  • Markets are noisy; past patterns may not repeat (regime change, wars, policy).
  • This is a course/research pipeline, not trading advice.


================================================================================
2. MODEL TYPE AND DESIGN CHOICES
================================================================================

What is an LSTM?
  A Long Short-Term Memory network processes sequences step by step and keeps a
  hidden state that summarises “what happened so far.”  It is a classic choice
  for ordered data when sequences are short (here, five days).

What does bidirectional mean?
  A forward LSTM reads day 1 → day 5; a backward LSTM reads day 5 → day 1.
  Their outputs are concatenated at each day.  The model can use context from
  both earlier and later days in the window when forming each position’s
  representation.  That helps when a late-window headline reinterprets earlier
  calm news.

What does attention pooling do?
  Instead of using only the last LSTM timestep, the model learns a weight
  alpha_t for each day t (non-negative, summing to 1).  The final vector is a
  weighted sum of day hidden states.  Larger alpha_t means “this day mattered
  more for the prediction.”  In evaluate_lstm reports, a heatmap shows these
  weights per test sample.

How to read attention weights:
  If day 4 has weight 0.45 and others are near 0.10, the classifier leaned on
  that day’s news aggregate.  Compare to which articles were present that day.

Why FinBERT (ProsusAI/finbert)?
  It is BERT pre-trained on financial text — tickers, earnings, macro language.
  Oil news shares vocabulary with that domain.  We use it as a frozen feature
  extractor so training time stays under ~30 minutes on CPU.

Why not train a full transformer from scratch?
  Transformers need large data and compute.  With ~1k articles and ~hundreds of
  windows, a from-scratch transformer would overfit and train slowly.  Frozen
  FinBERT + small LSTM is the pragmatic trade-off.

Why a projection layer (768 → lstm_hidden) before the LSTM?
  It cuts parameter count and speeds each epoch.  The LSTM operates in a
  128-dimensional space by default instead of 768.

Attention math (additive / Bahdanau-style)
  For each day t, LSTM output vector h_t (size lstm_hidden*2 if bidirectional).

    score_t = v^T · tanh(W · h_t)     (scalar)
    alpha_t = exp(score_t) / sum_j exp(score_j)   over days j in the window
    context = sum_t alpha_t · h_t     (vector)

  Learnable parameters: W (linear map), v (linear to scalar).  The MLP head
  maps context to 3 logits; softmax yields class probabilities.


================================================================================
3. DATA PIPELINE
================================================================================

Article storage (read-only)
  raw_articles/<MM_DD_YYYY>/article_<md5>.html
  Example: raw_articles/05_06_2026/article_abc123....html
  The scraper never modifies these files from the ML code path.

Parsed metadata (read-only)
  parsed_articles/*.json — titles and sources matched by URL hash for HTML
  filenames.  html_extractor may prepend a short metadata header to text.

Text extraction
  window_builder uses HTMLArticleExtractor (trafilatura with fallbacks) per file.
  Short or empty extractions are skipped.

FinBERT embedding (frozen)
  Each article: tokenise (max 256 tokens), run FinBERT, mean-pool token hidden
  states over non-padding positions → vector (768,).

Per-day embedding
  Up to max_articles_per_day articles (default 10), sorted filenames for
  reproducibility.  day_embedding = mean(article vectors).  If no articles:
  zero vector of length 768 (logged as zero-padded day).

Price labels (ground truth)
  price_fetcher downloads USO (or --ticker) via yfinance, caches CSV.
  log_return_t = ln(close_t / close_{t-1})
  If log_return > +flat_band_pct/100 → Up (2)
  If log_return < -flat_band_pct/100 → Down (0)
  Else → Flat (1)

Sliding window sample (worked example)
  window_days=5, gap_days=0, prediction_date = 2026-05-06:

    Window news days: 2026-05-01 … 2026-05-05 (five folders under raw_articles/)
    Label: direction of USO on 2026-05-06 from price table

  X shape (5, 768), y in {0,1,2}.  Samples are built for every prediction date
  where price labels exist and the window can be formed.

Embedding cache
  ml_model/outputs/embed_cache.pt stores { "YYYY-MM-DD": Tensor(768) } so
  reruns skip FinBERT for days already embedded.


================================================================================
4. TRAINING PROCEDURE
================================================================================

Why chronological splits?
  Training uses the oldest 80% of samples, validation the next 10%, test the
  newest 10%.  Shuffling before split would leak future news into training and
  inflate accuracy.  Time series must respect causality.

Class weighting
  When use_class_weights=True, CrossEntropyLoss uses weight_i = N_train / (3 *
  count_i) so rare classes (often Up or Down) contribute more per example.  That
  reduces the trivial strategy of always predicting Flat.

Early stopping
  Training stops if validation loss fails to improve for `patience` epochs (default
  6), restoring the best weights.  That limits overfitting when the model
  memorises train noise.

How to run training (from project root):

  pip install -r ml_model/requirements.txt

  python -m ml_model.train_lstm

  python -m ml_model.train_lstm --window 3 --gap 1 --epochs 20 --ticker USO

Outputs:
  ml_model/outputs/checkpoints/model_<timestamp>_<window>w_<gap>g.pkl
  ml_model/outputs/checkpoints/config_<timestamp>.json

Pickle payload keys: model_state_dict, config, train_history, embed_cache_path,
test_metrics.


================================================================================
5. EVALUATION AND BACKTESTING
================================================================================

Confusion matrix
  Rows = true class, columns = predicted.  Diagonal = correct.  Off-diagonal
  shows which mistakes dominate (e.g. predicting Flat when truth is Up).

Directional accuracy
  Fraction of test samples where predicted class equals actual.  Also reported
  per actual class (accuracy on Up days only, etc.) — important when Flat is
  common; overall accuracy can look high while Up/Down are never caught.

Backtest table
  evaluate_lstm backtest_vs_actual joins each test prediction_date with real
  closes, log return, probabilities, and correct flag.  This ties model output
  to price_fetcher ground truth, not pseudo-labels.

Walk-forward cross-validation (concept)
  Standard k-fold shuffles dates and leaks future into past.  Walk-forward
  trains on an expanding or rolling past window and tests on the next block
  forward in time.  Appropriate for time series; use it to see if metrics are
  stable across eras.  The default train_lstm script uses one chronological
  split; you can repeat training with different cutoff dates to approximate
  walk-forward manually.

Generate evaluation report:

  python -m ml_model.evaluate_lstm
  python -m ml_model.evaluate_lstm --checkpoint ml_model/outputs/checkpoints/model_....pkl

Writes ml_model/outputs/reports/eval_<timestamp>.html and .json.


================================================================================
6. CONFIGURATION PARAMETERS
================================================================================

All defaults live in pipeline_config.PipelineConfig.

Parameter              | Default              | Description
-----------------------|----------------------|------------------------------------------
window_days            | 5                    | Number of news days in each input sequence
gap_days               | 0                    | Days between last news day and prediction date
horizon_days           | 1                    | Reserved for multi-day targets (same-day label now)
flat_band_pct          | 0.5                  | Flat zone half-width in percent on log return
price_ticker           | USO                  | yfinance symbol for labels
price_start            | 2025-04-28           | Price history start
price_end              | 2026-05-13           | Price history end
embedding_model        | ProsusAI/finbert     | Frozen HuggingFace encoder
embed_dim              | 768                  | FinBERT hidden size
max_articles_per_day   | 10                   | Cap articles embedded per day
max_tokens_per_article | 256                  | Token cap per article
lstm_hidden            | 128                  | LSTM hidden units per direction
lstm_layers            | 2                    | Stacked LSTM depth (use 1 if slow)
dropout                | 0.3                  | Dropout in proj/LSTM/head
bidirectional          | True                 | Forward+backward LSTM
train_val_test_split   | (0.80, 0.10, 0.10)   | Chronological fractions
batch_size             | 16                   | Minibatch size
epochs                 | 30                   | Max epochs
lr                     | 2e-4                 | AdamW learning rate
weight_decay           | 1e-4                 | L2 regularisation
patience               | 6                    | Early stopping patience
use_class_weights      | True                 | Balance loss across classes
checkpoint_dir         | ml_model/outputs/... | Saved models
report_dir             | ml_model/outputs/... | HTML/JSON reports
embed_cache_path       | .../embed_cache.pt   | Day embedding cache
price_cache_path       | .../price_cache.csv  | Offline price CSV


================================================================================
7. HOW TO CHANGE THE PREDICTION WINDOW
================================================================================

CLI flags --window and --gap override pipeline_config for train, evaluate, predict.

  --window 3 --gap 0
    Three days of news; predict next trading day.  More samples, less context.

  --window 5 --gap 0
    Default: one trading week of news.

  --window 5 --gap 1
    News ends two days before prediction; simulates articles not yet available
    for the most recent day (production lag).

  --window 7 --gap 0
    Longer context; fewer valid samples at the start of the corpus.

  --window 3 --gap 2
    Short news history with two-day lag; useful stress-test for stale inputs.

When to use larger gap: live deployment where scraping or ETL finishes with delay.
When to use shorter window: very reactive markets or limited article coverage per day.


================================================================================
8. WALK-FORWARD CROSS-VALIDATION DETAILS
================================================================================

Fold construction (manual procedure with this codebase):
  1. Sort all window samples by prediction_date.
  2. Choose fold boundaries (e.g. four equal time blocks).
  3. For fold k: train on all samples before block k, test on block k.
  4. Record accuracy, macro F1, and per-class counts on each test block.

Why not standard k-fold?
  Random folds mix 2026-05-10 with 2026-04-29 in train and test, letting the
  model see “future” news patterns during training.  Walk-forward respects time.

Stratification check
  Report class counts per fold.  If a fold has zero Up days, accuracy is not
  comparable.  Always read per-class metrics alongside headline accuracy.

Parameter sensitivity (fill in after you run experiments):

  Config              | Test accuracy | Macro F1 | Notes
  --------------------|---------------|----------|---------------------------
  window=5 gap=0      | (run)         | (run)    | default
  window=3 gap=0      | (run)         | (run)    |
  window=5 gap=1      | (run)         | (run)    |
  window=7 gap=0      | (run)         | (run)    |
  window=3 gap=2      | (run)         | (run)    |


================================================================================
9. FILE INDEX
================================================================================

pipeline_config.py       — Dataclass of all hyperparameters and paths for LSTM pipeline.
data/price_fetcher.py    — Downloads USO prices; log returns; 3-class labels; CSV cache.
data/window_builder.py   — HTML → frozen FinBERT → daily vectors → sliding (X, y).
data/html_extractor.py   — Shared HTML text extraction (used by window_builder).
data/preprocessor.py     — Tokeniser utilities for the transformer stack (legacy path).
data/dataset.py          — PyTorch dataset for the multi-task transformer (legacy path).
model_lstm.py            — OilLSTMPredictor: BiLSTM + attention + 3-class head.
train_lstm.py            — CLI training: split, fit, checkpoint, test metrics.
evaluate_lstm.py         — Confusion matrix, backtest table, HTML/JSON reports.
predict_lstm.py          — Single-date inference with attention and article list.
config.py                — Original multi-task transformer configuration.
model.py                 — OilMarketTransformer (FinBERT fine-tune path, legacy).
run_training.py          — Entry point for transformer training (legacy).
run_inference.py         — Entry point for transformer inference (legacy).
training/                — Trainer, losses, metrics for transformer path.
inference/               — Predictor and report generator for transformer path.
modules/                 — Transformer building blocks (encoder, heads, attention).
utils/                   — Logging, seeds, parameter counting helpers.
requirements.txt         — Python dependencies including yfinance and sklearn.
data_snapshot.json       — Inventory of raw_articles file counts (informational).
ML_MODEL_README.txt      — This document.

outputs/checkpoints/     — Saved model_*.pkl and config_*.json (created at train time).
outputs/reports/         — eval_*.html / eval_*.json from evaluate_lstm.
outputs/embed_cache.pt   — Cached FinBERT day embeddings (created on first build).
outputs/price_cache.csv  — Cached yfinance prices (created on first label fetch).


================================================================================
QUICK COMMAND REFERENCE
================================================================================

  python -m ml_model.train_lstm
  python -m ml_model.evaluate_lstm
  python -m ml_model.predict_lstm --end-date 2026-05-10

================================================================================
