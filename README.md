# Distributed Google Scraper

A robust, distributed web scraping system designed to bypass anti-bot protections using GoLogin anti-detect profiles, Playwright stealth, and automated CAPTCHA solving.

## Core Components

### 1. `config.py`
The central configuration hub that manages the environment and scraper behavior:
- **Security**: Loads API keys (CapSolver, GoLogin) and proxy settings from `.env`.
- **Paths**: Manages project directories for `raw_html`, `search_results`, and `cache`.
- **Human Mimicry**: Defines the constants for randomized delays (Gaussian distribution) and scrolling behavior.
- **Logging**: Sets up system-wide logging for uniform debugging across nodes.

### 2. `scraper_engine.py`
The main execution engine responsible for the browser-level interactions:
- **GoLogin Integration**: Connects to GoLogin's browser instances via CDP (Chrome DevTools Protocol) to leverage high-quality browser fingerprints.
- **Playwright Stealth**: Uses `playwright-stealth` to further mask automation signatures.
- **CAPTCHA Bypass**: Integrates with CapSolver to automatically solve reCAPTCHAs encountered during Google searches.
- **Resilience**: Features automatic session restarts after a set number of tasks to flush memory and avoid detection.
- **Deduplication**: Maintains a local MD5 hash cache to prevent re-scraping the same URLs across different runs or nodes.

### 3. `run_scripts_in_parallel.py`
An orchestration layer for horizontal scaling on a local machine:
- **Node Management**: Launches multiple instances of the `scraper_engine.py` as background processes.
- **Load Balancing**: Automatically partitions the tasks in `tasks.json` across the available GoLogin profiles defined in the cluster configuration.
- **Process Supervision**: Captures individual node output into dedicated log files (`node_0.log`, `node_1.log`, etc.) and manages graceful shutdowns.

## Quick Start

1.  **Environment Setup**: Ensure a `.env` file exists with valid `GOLOGIN_API_TOKEN` and `CAPTCHA_API_KEY`.
2.  **Task Definition**: Place your target queries in `tasks.json`.
3.  **Launch Cluster**:
    ```bash
    python run_scripts_in_parallel.py
    ```
4.  **Monitoring**: View real-time logs for a specific node:
    ```bash
    tail -f node_0.log
    ```