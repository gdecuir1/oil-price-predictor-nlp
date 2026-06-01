# Oil Price Prediction via Financial News Scraping

An end-to-end NLP pipeline that scrapes financial news from Google News, extracts sentiment 
using FinBERT, and predicts oil price direction (Up / Flat / Down) with a BiLSTM + attention model.

> **Group Project** — Built collaboratively by [Gabrielle DeCuir](https://github.com/gdecuir1) 
> and [Kaleb Aklilu](https://github.com/kalk-ak) as a final project for Information Retrieval.

---

## Overview

Oil prices are among the most volatile signals in the global economy. Market-moving headlines 
often emerge hours or days before they appear in formal economic releases. This project builds 
a pipeline that harvests ~1,000 financial news articles from Google News, links each article 
to its corresponding trading-day oil price, and trains a model to predict the next day's 
price direction.

---

## How It Works

### 1. Data Collection
- 27 Boolean-style queries across: Oil & Energy, Macro/Fed, Geopolitics, Major Equities, Crypto, and Earnings
- Scraped daily from April 28 – May 12, 2026 → ~1,000 article URLs across 14 trading days
- Results stored as date-partitioned JSON batches (`tasks_YYYY-MM-DD.json`)

### 2. Anti-Detection Scraping Stack
Google aggressively blocks automated clients. The pipeline layers four defenses:

- **VPN** — masks institutional IP blocks
- **Rotating Residential Proxies** — assigns a fresh IP per search request
- **GoLogin Anti-Detect Browser** — unique browser fingerprint per session (user-agent, WebGL, canvas noise, fonts); profile restarts every k ~ N(5,2) tasks
- **Human Mimicry** — Gaussian-randomized delays + erratic scroll events to simulate real user behavior
- **CAPTCHA Solver** — CapSolver API automatically resolves reCAPTCHA challenges unattended

### 3. Parsing & Article Retrieval
- `parser.py` extracts metadata from Google News HTML using JSON-LD, OpenGraph, DOM selectors, and CMS heuristics
- `new_article_scraper.py` fetches full article HTML and stores it in date-partitioned folders (`raw_articles/YYYY-MM-DD/`)

### 4. Feature Engineering
- FinBERT converts each article into a 776-dim sentiment vector
- A projection layer compresses this to 256 dimensions to reduce noise

### 5. Model (v3 — BiLSTM + Attention)
- **Sliding window:** 5 days of news context → predict day 6 price direction
- **Bidirectional LSTM** captures how early-week headlines are reinterpreted by late-week news
- **Attention mechanism** weights days by relative news significance
- **Output:** 3-class classifier → Up / Flat / Down

> Earlier transformer-based attempts (v1, v2) overfit on the small dataset with only 15% 
> test accuracy. The BiLSTM generalized significantly better.

---

## Core Components

### `config.py`
Central configuration hub:
- Loads API keys (CapSolver, GoLogin) and proxy settings from `.env`
- Manages project directories for `raw_html`, `search_results`, and `cache`
- Defines constants for randomized delays and scrolling behavior
- Sets up system-wide logging

### `scraper_engine.py`
Main browser-level execution engine:
- Connects to GoLogin via CDP (Chrome DevTools Protocol)
- Uses `playwright-stealth` to mask automation signatures
- Integrates CapSolver for automatic CAPTCHA bypass
- Features session restarts after k tasks to flush memory
- Maintains MD5 hash cache to prevent duplicate scraping across runs

### `run_scripts_in_parallel.py` *(excluded from this repo)*
Orchestration layer for horizontal scaling:
- Launches multiple `scraper_engine.py` instances as background processes
- Partitions tasks across available GoLogin profiles
- Captures per-node output into dedicated log files (`node_0.log`, `node_1.log`, etc.)

---

## Results

| Mode | Accuracy |
|---|---|
| Binary (Up vs. Down) | 61% |
| 3-Way (Up / Flat / Down) | 55.56% |

Chronological train/val/test split: 80% / 10% / 10%

---

## Quick Start

1. **Environment Setup:** Create a `.env` file with valid `GOLOGIN_API_TOKEN` and `CAPTCHA_API_KEY`
2. **Task Definition:** Place your target queries in `tasks.json`
3. **Run scraper:**
```bash
python scraper_engine.py
```
4. **Monitor logs:**
```bash
tail -f node_0.log
```

---

## Tech Stack

- Python, Playwright, playwright-stealth, GoLogin SDK
- FinBERT, PyTorch
- yfinance (oil price ground truth via USO/CL=F)
- CapSolver API

---

## Limitations

- 14 trading days of data — model is sensitive to train/test split cutoff
- Data sparsity on weekends/holidays creates gaps in the sliding window
- Future work: expand to a 6-month collection window for stronger generalization

---

## Contributors

Built as a group final project for Information Retrieval:

- [Gabrielle DeCuir](https://github.com/gdecuir1)
- [Kaleb Aklilu](https://github.com/kalk-ak)
