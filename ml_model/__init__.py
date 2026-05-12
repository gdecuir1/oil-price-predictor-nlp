"""
Oil Market Prediction Transformer Model
========================================

A multi-task transformer neural network for predicting US oil stock market
movements based on scraped news article content. The model ingests raw HTML
articles, extracts and encodes textual features, and produces a comprehensive
prediction report across multiple sub-domains.

Primary prediction: Market direction (Up / Unchanged / Down)

Sub-domain predictions:
    - Price movement magnitude
    - Expected timeframe of movement
    - Market volatility assessment
    - Article sentiment analysis
    - Supply-side impact
    - Demand-side impact
    - Geopolitical risk level

Package structure:
    - config: Centralized hyperparameters and paths
    - data: HTML extraction, preprocessing, and PyTorch datasets
    - modules: Transformer building blocks (embeddings, attention, encoder, heads)
    - training: Training loop, loss functions, evaluation metrics
    - inference: Prediction pipeline and report generation
    - utils: Logging, helpers, and common utilities
"""

__version__ = "1.0.0"
__author__ = "Oil Market Prediction Team"
