# Bitcoin Attention with Google Trends Data

This repository studies the relationship between Bitcoin market dynamics and online attention.

The core idea is behavioral-finance oriented: search intensity from Google Trends can be treated as a proxy for public attention, then compared against Bitcoin volatility, trading volume, and returns using time-series methods.

The original notebook was reorganized into a reproducible Python project. Because the raw Bitcoin and Google Trends CSV files are not included in the public repository, the script generates a deterministic demo dataset with the same structure and runs the complete analysis pipeline.

## What the Project Does

- builds Bitcoin-style financial variables: log returns, realized volatility, log volume,
- builds an attention variable similar to Google Trends search interest,
- standardizes and log-transforms search attention into `log-SQ`,
- estimates Vector Autoregression (VAR) models,
- runs Granger causality tests,
- generates impulse response functions,
- performs a subsample analysis around a structural-break-style date.

## Why It Is Interesting

Bitcoin is not only a financial asset; it is also an attention-driven market. Public search behavior can rise around volatility, rallies, crashes, and media cycles. This project explores whether attention helps explain market variables, or whether market movements predict attention.

The project is useful as a compact example of:

- time-series preprocessing,
- behavioral finance,
- econometric modeling,
- VAR systems,
- Granger causality,
- impulse response analysis.

## Run

```bash
pip install -r requirements.txt
python bitcoin_attention_analysis.py
```

## Outputs

Running the script creates local CSV outputs under `outputs/` and figures under `figures/`.

Included example figures:

- `figures/bitcoin_attention_price.png`
- `figures/bitcoin_attention_correlation.png`
- `figures/bitcoin_attention_irf.png`

The generated CSV files are ignored by Git and can be recreated by rerunning the script.
