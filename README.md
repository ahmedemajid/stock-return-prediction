# Stock Return Prediction

Comparing OLS vs Ridge regression for predicting daily stock returns on 32 large-cap stocks (2020-2024).

## What I Built

Two approaches to predict next-day returns using technical features (daily returns, 30/90-day volatility, volume):

- OLS regression (implemented from scratch using NumPy's normal equation)
- Ridge regression (scikit-learn with cross-validated alpha selection)

Tested on 32 stocks across tech, finance, retail, and healthcare.

## Results

Both models overfit badly. Mean test R² was negative (-0.013 for OLS, -0.011 for Ridge).

Ridge regularization helped slightly - 24 out of 32 stocks showed improvement - but couldn't fix the fundamental problem: simple linear features don't predict daily returns well.

## Key Findings

Used chronological 80/20 train/test splits (trained on first 80% of dates, tested on last 20%). No shuffling to avoid lookahead bias.

Results showed:
- Training R² was positive (models fit historical data)
- Test R² went negative (models couldn't generalize)
- Each stock responded differently to the same features (no universal pattern)
- Ridge's regularization reduced coefficient magnitudes but couldn't create signal from noise

The negative test R² means the models performed worse than just predicting the mean. Daily returns are too noisy for these features.

## Running It
```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn
jupyter notebook SimpleStockAnalysis.ipynb
```

Data pulls from Yahoo Finance automatically.

## What I Learned

Proper validation matters - chronological splits revealed the models don't actually work, even though training metrics looked okay. Better to know a model doesn't work than to fool yourself with good in-sample metrics.
