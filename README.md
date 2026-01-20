# News_Sentiment_Analysis_on_Time_Series_Models_Undergraduate_CH21B058
Financial news sentiment analysis by generating embeddings using FinBERT and testing exogemeous response on Time Series models using the SET50 Thai Stock dataset.

# FinBERT-Based Financial News Sentiment and Stock Return Analysis

## Overview
This project studies whether **financial news sentiment**, extracted using a pretrained **FinBERT** model, improves short-term **stock return prediction**.  
News data is collected from Google News, sentiment is quantified using FinBERT, and its incremental predictive value is evaluated using both **statistical** and **deep learning** time-series models.

The analysis is conducted on **SET50 (Thailand) equities**, combining daily news sentiment with historical stock prices.

## Objectives
- Extract domain-specific sentiment from financial news using FinBERT
- Align news sentiment with daily stock returns
- Test whether sentiment improves predictive performance over price-only models
- Quantify improvement (or lack thereof) using objective error metrics

## Data Sources
### News Data
- Google News API
- Query format: `<Company Name> Thailand stock`
- Fields used:
  - headline
  - description
  - publication time

### Market Data
- Daily adjusted stock prices
- Downloaded using `yfinance`
- Two-year historical window per ticker
  
## Methodology

### 1. News Preprocessing
- Combined headline and description into a single text field
- Converted relative timestamps (e.g., “2 days ago”) into absolute dates
- Aggregated news at a daily frequency per ticker

### 2. Sentiment Analysis (FinBERT)
- Used pretrained `ProsusAI/finbert`
- Sentiment labels mapped to numerical scores:
  - Positive → +1
  - Neutral → 0
  - Negative → −1
- Computed daily average sentiment per stock

### 3. Feature Engineering
- Daily stock returns computed from adjusted close prices
- Missing-news days treated as neutral sentiment
- Final dataset merged on `(ticker, date)


## Models Evaluated

### Statistical Models
- **ARIMA (price-only)**
- **ARIMA with sentiment as an exogenous variable**

### Deep Learning Models
- **BiLSTM (price-only)**
- **BiLSTM with sentiment as an additional input feature**

All models were evaluated using an **80/20 time-based train-test split**.

## Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Directional Accuracy (sign of return prediction)

## Results & Key Findings

- Sentiment-augmented models showed **only marginal improvements** in predictive accuracy.
- MSE improvement was typically **~1–2%**, and only for a **small subset of tickers**.
- For most stocks, sentiment inclusion had **no statistically meaningful impact**.

## Why Sentiment Impact Was Limited

1. **Low Signal-to-Noise Ratio**  
   Daily stock returns are dominated by market microstructure noise, which overwhelms weak sentiment signals.

2. **Sparse and Uneven News Coverage**  
   Many stocks had few articles per day, reducing the reliability of daily sentiment averages.

3. **Pretrained Model Limitations**  
   FinBERT was not fine-tuned on Thai market–specific or company-specific news.

4. **Market Efficiency**  
   Public news is often priced in rapidly, limiting predictive power at daily horizons.

5. **Model Objective**  
   Minimizing MSE favors stable forecasts; sentiment effects may influence direction or volatility rather than point forecasts.

## Conclusion
This project demonstrates that while **FinBERT effectively extracts financial sentiment**, incorporating daily news sentiment provides **limited incremental predictive value** for short-horizon stock return forecasting.

The findings highlight the difficulty of converting qualitative news signals into consistent alpha and reinforce the importance of **rigorous empirical validation** when applying NLP models to financial markets.

## Tech Stack used
- Python
- GoogleNews API
- Hugging Face Transformers
- PyTorch
- yfinance
- statsmodels
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib
