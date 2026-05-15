================================================================================
ML_MODEL — LSTM Oil Direction Pipeline
================================================================================

This document describes the LSTM-based pipeline that predicts short-term US oil
price direction from sequences of scraped news articles.  It is written for two
audiences: readers new to machine learning (plain-language explanations) and
readers who want implementation and design detail.


================================================================================
0. SETUP — PREREQUISITES, DEPENDENCIES, AND HOW TO RUN
================================================================================

This section is for someone cloning the repo who needs to install packages and
run train_lstm, evaluate_lstm, or predict_lstm from the project root (the
directory that contains ml_model/ and raw_articles/).

Prerequisites (not installed by pip)

  • Python 3.10 or newer (3.11+ recommended).
  • A virtual environment is optional but recommended:
      python -m venv .venv
      source .venv/bin/activate          # macOS / Linux
      .venv\Scripts\activate             # Windows
  • Scraped article HTML under raw_articles/<MM_DD_YYYY>/article_<hash>.html.
    Training, evaluation, and prediction all read from this tree (read-only).
  • Optional: parsed_articles/*.json for richer metadata in extracted text.
  • Network access on first run:
      - yfinance downloads price history for the label ticker (default USO).
      - Hugging Face downloads ProsusAI/finbert (~400 MB) for article embeddings.
  • For evaluate_lstm and predict_lstm: at least one trained checkpoint
    ml_model/outputs/checkpoints/model_*.pkl (created by train_lstm).

Install Python dependencies (all three LSTM entry points use the same set)

  From the project root:

    pip install -r ml_model/requirements.txt

  Or install packages individually (minimum versions match requirements.txt):

    pip install "torch>=2.0.0" "transformers>=4.30.0" "numpy>=1.24.0" \
      "pandas>=2.0.0" "yfinance>=0.2.0" "scikit-learn>=1.3.0" \
      "beautifulsoup4>=4.12.0" "trafilatura>=1.6.0" "readability-lxml>=0.8.1" \
      "lxml>=4.9.0" "tqdm>=4.65.0"

Package list and what each is used for

  Package (pip name)     | Used by
  -----------------------|--------------------------------------------------
  torch                  | LSTM model, tensors, training/inference loops
  transformers           | Frozen FinBERT encoder (window_builder)
  numpy                  | Arrays, metrics, window tensors
  pandas                 | Price labels, dates, backtest tables
  yfinance               | Download/cache USO (or --ticker) prices
  scikit-learn           | Accuracy, F1, confusion matrix, classification report
  beautifulsoup4         | HTML text extraction fallback (html_extractor)
  trafilatura            | Primary HTML article extraction
  readability-lxml       | HTML extraction fallback
  lxml                   | Parser backend for readability / BeautifulSoup
  tqdm                   | Progress bars elsewhere in ml_model (optional for LSTM CLI)

  Standard library only (no pip install): argparse, json, logging, pickle,
  pathlib, datetime, dataclasses, etc.

  evaluate_lstm and predict_lstm do not require extra packages beyond the list
  above; they use the same data stack as train_lstm (FinBERT, HTML extraction,
  yfinance).

Typical first-time workflow

  1. pip install -r ml_model/requirements.txt
  2. Ensure raw_articles/ is populated (scraper or provided dataset).
  3. python -m ml_model.train_lstm
  4. python -m ml_model.evaluate_lstm
  5. python -m ml_model.predict_lstm --end-date YYYY-MM-DD

Artifacts created automatically on first use

  ml_model/outputs/embed_cache_v3.pt — cached FinBERT + keyword day vectors (v3)
  ml_model/outputs/price_cache.csv  — cached yfinance prices
  ml_model/outputs/checkpoints/     — model_*.pkl after training
  ml_model/outputs/reports/         — eval_*.html / eval_*.json after evaluation

GPU is optional; all scripts default to CPU if CUDA is unavailable.


================================================================================
1. WHAT THIS PROJECT DOES
================================================================================

The system answers a focused question: after reading oil-related news from
several consecutive calendar days, will the oil market go up, stay roughly flat,
or go down on the next trading day we care about?

Inputs are not prices during the window — they are the *text* of news articles
saved as HTML under raw_articles/.  Each article is converted to a numerical
embedding (a list of 768 numbers from FinBERT).  For each day in the window we
average embeddings from up to twenty articles (FinBERT [CLS] per article in v2).
  Those daily vectors form a short
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
2. MODEL TYPE AND DESIGN CHOICES (v3 — recommended)
================================================================================

IMPORTANT: Checkpoints are not interchangeable across v1 / v2 / v3.  After
upgrading, retrain with:

    python -m ml_model.train_lstm

Compare old vs new models:

    python -m ml_model.compare_checkpoints

What v3 fixes (after v2 test accuracy ~15%)
  • **Smaller LSTM** again: 128 hidden, 2 layers, proj 256 (~500k params vs ~5.5M).
  • **Keyword features** (8 groups: bullish, bearish, supply, demand, geo, etc.)
    concatenated to each day vector — see data/keyword_extractor.py.
  • **Binary labels (default)**: label_mode=binary drops Flat days; only Down vs Up.
    Avoids the “model predicts Flat but test has no Flat” failure mode.
  • **Ternary option**: label_mode=ternary with flat_band_pct=0.35 (narrower band
    than 0.5 → fewer Flat labels than before).
  • Masked attention, CLS pooling, feature normalisation, weighted sampling.
  • **Mild class balance**: `class_weight_mode=sqrt` + sqrt weighted sampler (not full inverse).
  • Early stopping on **val_min_recall** (worst-class recall); skips collapsed val epochs.
    Stops if Down or Up val recall is 0 for 8 consecutive epochs (after min_train_epochs=25).
  • **Focal loss** (gamma=2), **FinBERT-only** z-score norm (keywords keep raw scale).
  • Threshold tuning rejects cutoffs that predict only one class on val.
  • **Down vs Up focus** metrics printed in train/eval when ternary is used.
  • **Decision threshold (B):** after training, sweep P(Up) on validation; stored as
    up_probability_threshold in checkpoint (see PERFORMANCE_REPORT.txt for recent runs).
  • **Regularisation (C):** mlp_hidden=48, dropout=0.35, weight_decay=1e-3, lr=1e-4.

Embeddings cache: ml_model/outputs/embed_cache_v3.pt (768 FinBERT + 8 keywords = 776 dims).

Labels and horizon (D)
  • Default label_mode=binary drops Flat days — recommended unless you need an
    “unchanged” class (then ternary with flat_band_pct=0.35).
  • Prediction horizon: news through trading day T−1 → USO move on day T
    (prediction_date in metadata).  Backtest/eval reports label this explicitly.

Data and evaluation (E, F)
  • More article days under raw_articles/ is the largest real lever.
  • python -m ml_model.inspect_keywords — mean keyword groups by Down/Up label.
  • Report balanced accuracy and per-class recall; ~23 test points are indicative only.
  • python -m ml_model.compare_checkpoints — side-by-side bal_acc / macro F1;
    older 3-class checkpoints may still beat a collapsed binary run.

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

Why a projection layer (768 → proj_dim → lstm_hidden) before the LSTM?
  FinBERT vectors are high-dimensional; a learned bottleneck lets the LSTM focus
  on compressed temporal patterns.  Default proj_dim=512, lstm_hidden=256.

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
  Each article: tokenise (max 256 tokens), run FinBERT.  Default article_pooling
  is "cls": use the [CLS] token embedding (index 0).

Keyword features (v3)
  For each article, count normalised hits in 8 oil-domain keyword groups
  (bullish, bearish, supply_up, supply_down, demand_up, demand_down,
  geopolitical, volatility).  Day-level keyword vector = mean over articles.
  Concatenated: day_vector = [FinBERT_768 | keywords_8] → input_dim 776.

Per-day embedding
  Up to max_articles_per_day articles (default 20).  If no articles: zero vector;
  LSTM attention mask ignores those days.

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

Label modes
  binary (default): training samples only on days USO moved Up or Down; Flat
  days are skipped.  Targets are 0=Down, 1=Up.
  ternary: 0=Down, 1=Flat, 2=Up using flat_band_pct on log returns.

Embedding cache
  ml_model/outputs/embed_cache_v3.pt stores { "YYYY-MM-DD": Tensor(776) }.


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
  Training stops if the chosen metric fails to improve for `patience` epochs
  (default 10).  Default early_stopping_metric is val_macro_f1 so the saved
  checkpoint favours balanced Up/Flat/Down performance, not only low loss.

Feature normalisation
  When normalize_features=True, each embedding dimension is z-scored using
  mean and std from the training split only.  feature_mean and feature_std are
  stored in the .pkl so predict_lstm and evaluate_lstm apply the same transform.

Weighted sampling
  use_weighted_sampler=True oversamples minority direction classes each epoch,
  complementing use_class_weights in the loss.

How to run training (from project root)

  See section 0 for install and prerequisites.  Then:

  python -m ml_model.train_lstm

  python -m ml_model.train_lstm --window 3 --gap 1 --epochs 20 --ticker USO

  python -m ml_model.compare_checkpoints

  CLI flags: --window, --gap, --epochs, --ticker (override pipeline_config defaults).
  Edit pipeline_config.py for label_mode ("binary" | "ternary"), use_keywords, etc.

Outputs:
  ml_model/outputs/checkpoints/model_<timestamp>_<window>w_<gap>g.pkl
  ml_model/outputs/checkpoints/config_<timestamp>.json

Pickle payload keys: model_state_dict, config, train_history, embed_cache_path,
test_metrics, feature_mean, feature_std.


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

How to run evaluation (evaluate_lstm)

  Prerequisites: same pip install and data as section 0; a checkpoint from
  train_lstm (or pass --checkpoint explicitly).

  From the project root:

    python -m ml_model.evaluate_lstm

  Uses the newest model_*.pkl in ml_model/outputs/checkpoints/ if --checkpoint
  is omitted.  Rebuilds test windows with FinBERT (reuses embed_cache.pt when
  possible) and compares predictions to held-out chronological test dates.

  Examples:

    python -m ml_model.evaluate_lstm --checkpoint ml_model/outputs/checkpoints/model_20260514_120000_5w_0g.pkl
    python -m ml_model.evaluate_lstm --window 3 --gap 1

  CLI flags: --checkpoint PATH, --window INT, --gap INT (window/gap must match
  how the checkpoint was trained unless you intentionally experiment).

  Outputs (created under ml_model/outputs/reports/):

    eval_<timestamp>.html  — confusion matrix, backtest table, attention heatmap
    eval_<timestamp>.json  — same metrics in machine-readable form


================================================================================
5B. SINGLE-DATE INFERENCE (predict_lstm)
================================================================================

predict_lstm scores one prediction date: it loads a checkpoint, builds the
news window ending before that date (per window_days and gap_days), runs the
BiLSTM, and prints the predicted class, probabilities, attention weights, and
article filenames per day.

Prerequisites

  Same dependencies as section 0 (pip install -r ml_model/requirements.txt).
  Requires raw_articles/ folders for each day in the window.  Requires a
  trained checkpoint (newest model_*.pkl by default).

How to run (from project root)

  python -m ml_model.predict_lstm --end-date YYYY-MM-DD

  --end-date is the prediction date (the day whose oil direction is predicted).
  News days are computed backward from that date using window_days and gap_days
  in the checkpoint config (overridable with --window / --gap).

  Examples:

    python -m ml_model.predict_lstm --end-date 2026-05-10
    python -m ml_model.predict_lstm --end-date 2026-05-10 --checkpoint ml_model/outputs/checkpoints/model_20260514_120000_5w_0g.pkl
    python -m ml_model.predict_lstm --end-date 2026-05-10 --window 5 --gap 0

  CLI flags: --end-date (required), --checkpoint PATH, --window INT, --gap INT.

  Output is printed to the terminal (class name, probabilities, attention per
  day, article paths).  If price data exists for that date, actual direction
  and match YES/NO are shown; dates beyond price_cache or non-trading days may
  have no ground-truth label.


================================================================================
6. CONFIGURATION PARAMETERS
================================================================================

All defaults live in pipeline_config.PipelineConfig.

Parameter              | Default              | Description
-----------------------|----------------------|------------------------------------------
label_mode             | binary               | binary (Down/Up only) or ternary
flat_band_pct          | 0.35                 | Flat band for ternary labels (%)
window_days            | 5                    | News days per sample
gap_days               | 0                    | Gap before prediction date
use_keywords           | True                 | Append 8 keyword features per day
finbert_dim            | 768                  | FinBERT vector size
input_dim              | 776                  | finbert_dim + keyword_dim
max_articles_per_day   | 20                   | Articles embedded per day
article_pooling        | cls                  | cls or mean per article
proj_dim               | 256                  | Projection width
lstm_hidden            | 128                  | LSTM hidden size
lstm_layers            | 2                    | LSTM depth
mlp_hidden             | 48                   | Classifier hidden size
dropout                | 0.35                 | Dropout rate
lr                     | 1e-4                 | AdamW learning rate
weight_decay           | 1e-3                 | AdamW L2
use_focal_loss         | True                 | Focal loss gamma=2 (hard examples)
normalize_finbert_only | True                 | Z-score FinBERT dims only; keywords raw
min_train_epochs       | 25                   | Before zero-recall early stop applies
tune_up_threshold      | True                 | Sweep P(Up) on val after training (binary)
threshold_tuning_metric| balanced_accuracy    | or macro_f1
up_probability_threshold| (set at train)      | Saved in .pkl; used at eval/predict
epochs                 | 50                   | Max training epochs
patience               | 12                   | Early stopping patience (non-collapsed metric)
early_stopping_metric  | val_min_recall       | Worst per-class val recall; also macro_f1, bal_acc, loss
zero_up/down_recall_patience | 8            | Stop if either class val recall is 0
class_weight_mode      | sqrt                 | none | sqrt | full
use_weighted_sampler   | True                 | sqrt-weighted oversampling in training
reject_collapsed_val   | True                 | Do not early-stop on all-Down or all-Up val epochs
threshold_min_class_recall | 0.15           | Threshold sweep must hit both classes on val
embed_cache_path       | .../embed_cache_v3.pt| Day vector cache
compare_checkpoints    | (script)             | python -m ml_model.compare_checkpoints
PERFORMANCE_REPORT.txt | (doc)                | Recent training runs and test metrics summary


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
  data/keyword_extractor.py — Oil-domain keyword hit features per article/day.
  data/window_builder.py   — HTML → FinBERT + keywords → sliding (X, y); binary filter.
  compare_checkpoints.py   — Side-by-side test metrics for two .pkl checkpoints.
  threshold_tuning.py    — Post-train P(Up) cutoff on validation (binary).
  inspect_keywords.py    — Mean keyword features by Down/Up label.
  PERFORMANCE_REPORT.txt — Summary of recent v3 binary training runs and analysis.
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

  # One-time setup (project root)
  pip install -r ml_model/requirements.txt

  # Train → evaluate → predict
  python -m ml_model.train_lstm
  python -m ml_model.evaluate_lstm
  python -m ml_model.compare_checkpoints
  python -m ml_model.inspect_keywords
  python -m ml_model.predict_lstm --end-date 2026-05-10

  # Recent run metrics (written after training sessions):
  #   ml_model/PERFORMANCE_REPORT.txt

  # Optional: point to a specific checkpoint
  python -m ml_model.evaluate_lstm --checkpoint ml_model/outputs/checkpoints/model_<timestamp>_5w_0g.pkl
  python -m ml_model.predict_lstm --end-date 2026-05-10 --checkpoint ml_model/outputs/checkpoints/model_<timestamp>_5w_0g.pkl

================================================================================
