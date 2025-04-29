
# Abstract
This project implements a regime-based algorithmic trading strategy for the Vietnamese VN30F futures market on a 1-minute timeframe. Using a Markov Switching Model, the system dynamically classifies the market into two distinct volatility regimes: high volatility and low volatility.

For each regime, a tailored set of technical indicators is used to generate trading signals:

In high volatility, a momentum-based strategy captures trend continuations using indicators such as Acceleration, ADX, MACD, Bollinger Bands breakouts, On-Balance Volume, and volume surges.

In low volatility, a mean reversion strategy exploits price oscillations using SMA Gap, Stochastic Oscillator, Williams %R, Commodity Channel Index, Bollinger Bands reversion, and volume capitulation signals.

Positions are initiated when at least 80% regime-specific conditions are met, increasing signal opportunities while maintaining robustness. The framework is modular, allowing easy customization of indicators, regime detection methods, and signal logic.

This approach aims to adapt trading behavior to changing market conditions, improving performance stability and risk management in a high-frequency environment.

# Backtesting and Optimization System
This project presents a modular, systematic trading strategy framework that integrates dynamic position sizing, robust risk management, and parameter optimization. The system simulates real-world trading conditions using historical data while adjusting trade sizes based on volatility and signal strength. It also offers comprehensive performance evaluation via various metrics and visualizations.

---

## Installation
- Requirement: pip
- Install the dependencies by:
```
pip install -r requirements.txt
```

# Related Work (Background)

Ang, A., & Timmermann, A. (2012).
Regime Changes and Financial Markets.
Annual Review of Financial Economics, 4(1), 313–337.
➔ A comprehensive review of how regime-switching models are applied in finance to explain volatility, returns, and asset allocation strategies.

Lo, A. W., & MacKinlay, A. C. (1997).
The Econometrics of Financial Markets.
Princeton University Press.
➔ This textbook discusses technical indicators, momentum strategies, and econometric models applied to trading — including empirical performance under changing market conditions.

Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).
Time Series Momentum.
Journal of Financial Economics, 104(2), 228–250.
➔ Empirical evidence supporting the success of momentum strategies, particularly in assets showing strong trend persistence during high volatility periods.

S&P Dow Jones Indices. (2021).
Understanding Mean Reversion in Volatility.
➔ A white paper discussing how volatility tends to mean revert, and how strategies can exploit low-volatility environments for short-term mean-reversion opportunities.


## Trading (Algorithm) Hypotheses

The design of this trading algorithm is based on the following core hypotheses:

Market volatility exhibits regime-switching behavior. (Using 5 minutes timeframe for less noise and clearer regime)
The VN30F futures market alternates between distinguishable high-volatility and low-volatility states. These regimes can be dynamically identified using a Markov Switching Model, allowing the trading strategy to adapt in real-time.

Different market regimes favor different trading styles.

In high volatility, price movements are more likely to persist in their direction, favoring momentum-based strategies that ride strong trends.

In low volatility, prices tend to oscillate around a mean value, favoring mean reversion strategies that exploit temporary price extremes.

Technical indicators can effectively capture regime-specific opportunities.
A carefully selected set of technical indicators—such as ADX, MACD, Bollinger Bands, Stochastic Oscillator, and Williams %R—can provide statistically significant signals for both trend continuation and mean reversion, when interpreted in the context of the prevailing volatility regime.

Volume dynamics enhance signal validity.
Volume-based indicators, such as On-Balance Volume and relative volume surges, add valuable information about market participation and conviction, improving the accuracy of momentum breakouts and mean reversion points.

Short-term intraday strategies can benefit from volatility-aware adaptation.
On a 1-minute timeframe, rapid shifts between trending and ranging conditions make regime awareness particularly critical. A static strategy is likely to underperform compared to a dynamic, volatility-responsive system.


# Data

## Data Source
- Historical market data including OHLC (Open, High, Low, Close) prices, trading volumes (from SSI fast connect data API).
More detail of SSI API can be found [here](https://guide.ssi.com.vn/ssi-products/)
- Data is sourced from reliable financial providers and stored in CSV format.


## Data Type
- Time series data
- Price and volume information (1 minute timeframe and 5 minutes timeframe)
- Computed technical indicators (SMA, ATR, RSI, ADX,...)

## Data Period
- **In-Sample:** e.g., January 2020 to December 2023
- **Out-of-Sample:** e.g., January 2024 to Now
---

## Data Processing

Raw data is cleaned and pre-processed to calculate essential technical indicators. This step includes:
- Calculating Technical Analysis Value
- Merging various data sources (price, volume, and index data).
- Aligning time series data for accurate analysis.
---

# Implementation

## Brief Implementation Description
The trading strategy is implemented in Python and consists of the following modules:
- **Backtesting Engine:** Simulates trade execution, dynamic position management (including partial exits and trailing stops), and performance metric calculation.
- **Optimization Module:** Uses Optuna for parameter tuning to maximize strategy performance. Also, using Hidden Markov Model for fitting a ML-based Regime detection
- **Result & Metric Analysis:** Provides visualization and quantitative analysis of performance (e.g., cumulative PNL, Sharpe Ratio, Maximum Drawdown, Win Rate, contract counts).

## Environment Setup and Replication Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/tysunqua/Markov-Switching-Scalping-Strategy.git
# In-sample Backtesting

## Description
In-sample backtesting is the process of calibrating and validating the trading strategy using historical data from the training period. This step helps fine-tune the model parameters and assess performance before applying the strategy to new data.  


## Parameters
- Strategy parameters (e.g., ADX window length, EMA window length,...)
- Regime Detection Model (e.g., Hidden Markov Model)
- Risk management settings (e.g., stop-loss, take-profit thresholds)
- Initial asset value (e.g., 10,000)

# Optimization

The optimization step involves fine-tuning the trading strategy parameters to maximize performance, measured by cumulative PNL. This step employs advanced hyperparameter tuning techniques to systematically search the parameter space.  

## Optimization Process / Methods / Library
- **Library:** [Optuna](https://optuna.org/)
- **Method:** Tree-structured Parzen Estimator (TPE) sampler
- **Process:** Multiple trials are conducted to evaluate various parameter combinations, selecting the best based on the objective function (maximizing cumulative PNL).

# Hidden Markov Model 
- **Library:** [Hmm](https://hmmlearn.readthedocs.io/en/latest/tutorial.html)
- **Method:** Maximize full covariance
- **Feature** ATR, ATR percentage change
- **Process:** Multiple trials are conducted to evaluate various parameter combinations, selecting the best based on the objective function (maximizing likelihood function).

## Parameters to Optimize
    - sma_window_length
    - sma_gap
    - momentum_lookback
    - acceleration_threshold
    - short_acceleration_threshold
    - take_profit_threshold_high_volatility
    - take_profit_threshold_low_volatility
    - cut_loss_threshold_high_volatility
    - cut_loss_threshold_low_volatility
    - quantity_window
    - quantity_multiply
    - short_extra_profit
    - rsi_window
    - rsi_threshold
    - ema_fast_period
    - ema_slow_period
    - adx_window
    - adx_threshold
    - macd_window
    - bb_window
    - obv_window
    - williams_r_window
    - williams_r_threshold
    - cci_window
    - cci_threshold
    - stochastic_oscillator_window
    - stochastic_oscillator_threshold

## Hyper-parameters of the Optimization Process
- **Number of Trials:** 1000
- **Sampler Seed:** 42  
- **Study Direction:** Maximize cumulative PNL

## HMM Optimization Process
- **Number of Trials:** 1000
- **Study Direction:** Maximize likelihood function of Covariance

## Optimization Result
The best parameter set is presented in a summary table with corresponding performance metrics.  

The results is stored in optimization/sma.db
The best parameters is stored in ```optimization/best_params.json``` file
```
{
    "sma_window_length": 151,
    "sma_gap": 0.09130296520194026,
    "momentum_lookback": 5,
    "acceleration_threshold": 0.14674352542981076,
    "short_acceleration_threshold": 0.09696341838143158,
    "take_profit_threshold_high_volatility": 5.5,
    "take_profit_threshold_low_volatility": 4.0,
    "cut_loss_threshold_high_volatility": 2.0,
    "cut_loss_threshold_low_volatility": 1.5,
    "quantity_window": 49,
    "quantity_multiply": 1,
    "short_extra_profit": 1.3383390481427497,
    "rsi_window": 25,
    "rsi_threshold": 41,
    "ema_fast_period": 2,
    "ema_slow_period": 13,
    "adx_window": 13,
    "adx_threshold": 36,
    "stochastic_oscillator_window": 97,
    "stochastic_oscillator_threshold": 14,
    "bb_window": 56,
    "cci_window": 29,
    "cci_threshold": 76,
    "williams_r_window": 10,
    "willr_threshold": 39
}
```
---

# Result
## Insample Result
**Cumulative PnL**

![Insample Return](doc/figure_2.png)

| **Metric**               | **Momentum Strategy** | **Markov-Switching Strategy** |
|--------------------------|-----------------------|-------------------------------|
| Sharpe Ratio             | 0.1352                | 0.1244                        |
| Maximum Drawdown         | 9.02%                 | 7.21%                         |
| Win Rate                 | 40.41%                | 40.31%                        |
| Final PNL                | 15,735.41             | 16,799.46                     |
| Total Long Trades        | 9,727                 | 15,330                        |
| Total Short Trades       | 10,270                | 15,482                        |


## Outsample Result
**Cumulative PnL**

![Outsample Return](doc/figure_1.png) 

| **Metric**               | **Momentum Strategy** | **Markov-Switching Strategy** |
|--------------------------|-----------------------|-------------------------------|
| Sharpe Ratio             | 0.0884                | 0.0796                        |
| Maximum Drawdown         | 41.80%                | 36.82%                        |
| Win Rate                 | 35.73%                | 40.31%                        |
| Final PNL                | 3,169.54              | 3,200.29                      |
| Total Long Trades        | 3,558                 | 5,591                         |
| Total Short Trades       | 3,300                 | 5,182                         |


# Conclusion

Considering both in-sample (first tuple) and out-of-sample (second tuple) performance, the Markov‐Switching strategy clearly outperforms the pure momentum approach on key risk and return metrics. While momentum edges out slightly on Sharpe ratio (0.1352 vs. 0.1244 in-sample; 0.0884 vs. 0.0796 out-of-sample), Markov‐Switching cuts maximum drawdown by over 20% in-sample (7.21% vs. 9.02%) and by nearly 12% out-of-sample (36.82% vs. 41.80%), delivers higher net profits both in-sample (≈ 16,799 vs. 15,735) and out-of-sample (≈ 3,200 vs. 3,170), and maintains at least comparable win rates (40.31% vs. 40.41% in-sample; 40.31% vs. 35.73% out-of-sample). Moreover, it captures significantly more trading opportunities, reflecting its adaptability to different market regimes. Overall, the Markov‐Switching framework provides a more resilient balance of profitability and risk control, making it the superior choice for both calibration and real-world deployment.

---

# References

- Zhang, M., Jiang, X., Fang, Z., Zeng, Y., & Xu, K. (2019). High-order Hidden Markov Model for trend prediction in financial time series. Physica A: Statistical Mechanics and its Applications, 517, 1–12. https://doi.org/10.1016/j.physa.2018.10.053​
- An article on implementing Hidden Markov Models in high-frequency trading via the MQL5 community:
https://www.mql5.com/en/articles/16830
---