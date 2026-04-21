# Machine Learning for Asset Pricing  
### A Replication-Oriented Study Based on *Empirical Asset Pricing via Machine Learning*


## Project Introduction

This project is based on the paper *Empirical Asset Pricing via Machine Learning* (Gu, Kelly, and Xiu, 2020), which shows that machine learning models—especially nonlinear ones—can outperform traditional linear regression in predicting stock returns by capturing complex interactions in the data.

In particular, the paper highlights that nonlinear models are able to extract predictive signals from high-dimensional financial data more effectively than linear models.

Reading this paper made me curious about whether these results would still hold in a much simpler and more constrained setting.

In this project, I focus on understanding when and why nonlinear models do not outperform linear models in financial return prediction, even though they are more flexible.

To explore this, I implemented a simplified cross-sectional asset pricing framework and compared several models (OLS, Elastic Net, Random Forest). I also looked at how things like the number of stocks and feature complexity affect performance in a noisy financial environment.


## Project Motivation

My interest in applying machine learning to financial markets led me to explore how these models perform in noisy environments.

While machine learning models have shown strong performance in many domains, their effectiveness in financial markets remains unclear. Financial return prediction is characterized by a low signal-to-noise ratio and unstable patterns over time, making it difficult for even flexible models to extract reliable signals.

This project is motivated by this question:  
**why do powerful machine learning models often fail to deliver consistent improvements in asset pricing tasks?
**
To explore this question, I implemented a representative asset pricing framework and examined under what conditions machine learning models can provide meaningful predictive power.


## Current Scope

The current implementation focuses on building a simplified but structured asset pricing pipeline:

- S&P 500 universe construction using a cached ticker list
- Price data loading through `yfinance`
- Feature engineering based on:
  - momentum
  - volatility
  - liquidity proxy
  - reversal
  - interaction terms
- Cross-sectional sampling of 100 stocks per date
- Cross-sectional normalization
- Model comparison across:
  - OLS
  - Elastic Net
  - Random Forest
- Evaluation using:
  - out-of-sample R²
  - long-short portfolio Sharpe ratio


## Pipeline Overview

The current pipeline follows a simplified cross-sectional asset pricing framework:

1. Load S&P 500 tickers from CSV  
2. Download price data  
3. Construct returns and prediction target  
4. Generate features (momentum, volatility, liquidity, interactions)  
5. Form a cross-sectional universe (100 stocks per date)  
6. Apply cross-sectional normalization (rank transformation)  
7. Train predictive models (OLS, Elastic Net, Random Forest)  
8. Evaluate out-of-sample performance via return prediction and long-short portfolio construction


## Current Results

The following results are obtained under the improved setup:

| Model | OOS R² | Sharpe |
|------|--------|--------|
| OLS  | 3.0326408857583864e-05 | 1.0770714509118666 |
| Elastic Net | 5.30899076840452e-05 | 1.0891526215068394 |
| Random Forest | -8.10239974979865e-05 | 0.531850431028678 |


## Experiment: Effect of Data Structure and Feature Complexity

To investigate the performance of machine learning models in financial return prediction, I designed an experiment focusing on how data structure and feature complexity affect model behavior.

### Baseline Setup

In the baseline setting, I used:

- a fixed universe of manually selected tickers  
- a relatively simple feature set without interaction terms  

This setup provides a controlled environment with limited cross-sectional diversity and lower model complexity.

---

### Hypothesis

I hypothesized that the limited performance of nonlinear models (e.g., Random Forest) could be due to:

- insufficient cross-sectional diversity  
- lack of feature interactions necessary to capture nonlinear relationships  

---

### Improved Setup

To test this hypothesis, I modified the data construction as follows:

- replaced the fixed universe with cross-sectional sampling  
  (randomly selecting 100 stocks per date from the S&P 500 universe)  
- expanded the feature set by including interaction terms  

These changes aim to provide richer and more complex feature space for nonlinear models.

---

### Result

The following table compares the baseline and improved setups:

| Model | OOS R² (Baseline) | OOS R² (Improved) | Sharpe (Baseline) | Sharpe (Improved) |
|------|------------------:|------------------:|------------------:|------------------:|
| OLS  | 0.000051 | 0.000030 | 0.650 | 1.069 |
| ENET | 0.000171 | 0.000053 | 1.181 | 1.094 |
| RF   | -0.000320 | -0.000087 | 0.655 | 0.593 |

Overall, the results are mixed:

- OLS shows a clear improvement in Sharpe ratio  
- Elastic Net remains relatively stable  
- Random Forest does not improve and slightly underperforms compared to the baseline  

### Summary

The results show that increasing feature complexity and cross-sectional diversity does not lead to consistent improvements in model performance.

While linear models remain relatively stable, nonlinear models such as Random Forest do not exhibit clear gains and, in some cases, underperform.


## Key Observations

The results show that model performance in financial return prediction is quite sensitive to how the data is structured.

In particular, increasing feature complexity and cross-sectional diversity does not necessarily lead to better performance for nonlinear models. Even though models like Random Forest are more flexible, they do not consistently outperform linear models in this setting.

A likely reason is the low signal-to-noise ratio in financial data. In such environments, more flexible models tend to fit noise rather than capture stable patterns, which leads to unstable out-of-sample performance.

This helps explain why simpler linear models remain competitive, as they are less prone to overfitting and better suited for weak and noisy signals.


## Limitations

This project has several limitations related to data, feature design, and temporal structure.

First, the dataset is relatively small compared to that used in the original paper.  
In a low signal-to-noise environment like financial markets, insufficient data can make it difficult for models (especially nonlinear models) to learn stable and meaningful patterns.

Second, the feature set is still limited.  
Although interaction terms were introduced, the overall features may not be rich enough to fully capture complex relationships that more flexible models rely on.

Third, the current setup does not fully use time information.  
Although the model is trained on data from multiple time periods, it is not updated over time using a rolling or expanding approach.  
Because of this, it may not fully capture how relationships change over time or reflect realistic prediction performance.

Overall, these limitations suggest that the current results should be interpreted as a simplified exploration of the framework, rather than a complete replication.


## Next Steps

From the current results, the next step is to improve both the data setup and the modeling approach.

One limitation of the current setup is that it does not fully use information across time.  
Because of this, it is difficult to capture time-varying patterns in financial data.

Introducing a rolling or expanding training framework would help the model learn more stable relationships.  
This also aligns with the idea that financial data often have strong time-series characteristics.

I also plan to try more flexible models such as neural networks,  
to see whether they can better capture nonlinear relationships.

Overall, the goal is to better understand when machine learning models can actually provide meaningful improvements in asset pricing.


## Repository Structure
'''
ml_asset_pricing/
├── data/
│   ├── sp500_tickers_2026.csv
│   └── raw/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── train.py
│   └── models/
│       ├── linear.py
│       ├── elastic_net.py
│       └── tree.py
├── notebooks/
└── README.md '''

