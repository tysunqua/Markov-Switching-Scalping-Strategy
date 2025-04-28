
# Abstract
This project presents a modular backtesting framework for systematic trading strategies that integrates dynamic position sizing, robust risk management, and parameter optimization. By leveraging technical indicators like ATR, RSI, and SMA, the system generates trading signals and adjusts trade sizes based on market volatility and signal strength. With built-in mechanisms for stop-losses, take-profits, partial exits, and trailing stops, it ensures effective risk management and capital preservation. An Optuna-based optimization module fine-tunes parameters to maximize performance, making the framework a powerful and flexible tool for both research and practical trading applications.

# Introduction
In today's dynamic financial markets, systematic trading strategies require robust frameworks that adapt to changing conditions while effectively managing risk. This strategy combines technical analysis with dynamic position sizing and rigorous risk management to capture profitable trading opportunities. By leveraging key indicators such as moving averages, ATR, and RSI, the strategy identifies entry and exit signals that reflect market momentum and volatility.
At its core, the strategy uses dynamic position sizing to adjust the number of contracts based on signal strength and market volatility, ensuring that exposure is aligned with the prevailing risk environment. Integrated risk management tools—such as stop-losses, take-profit levels, partial exits, and trailing stops—help protect capital and lock in gains. Furthermore, the inclusion of an optimization module using Optuna allows for the fine-tuning of strategy parameters, enhancing performance and adaptability across different market conditions.
Overall, this approach provides a comprehensive and systematic framework that bridges theoretical trading models with real-world applications, offering traders a disciplined method to navigate volatile markets while aiming for consistent performance.

# Backtesting and Optimization System
This project presents a modular, systematic trading strategy framework that integrates dynamic position sizing, robust risk management, and parameter optimization. The system simulates real-world trading conditions using historical data while adjusting trade sizes based on volatility and signal strength. It also offers comprehensive performance evaluation via various metrics and visualizations.

## Feature
- [x] Research the 1-minute candle scalping to beat the fee and spread (0.47 in total)
- [x] Validate Test Case: Momentum Signal, Reversion Signal and Backtesting
- [x] Optimize parameters
- [x] Evaluate backtesting and optimization
- [ ] Paper trade

---

## Installation
- Requirement: pip
- Install the dependencies by:
```
pip install -r requirements.txt
```

# Related Work (Background)

Many academic studies and industry reports have explored quantitative trading strategies using technical indicators such as moving averages, ATR, and RSI. Prior work shows that dynamic position sizing—where trade sizes adapt to market volatility and signal strength—can improve risk-adjusted returns.

**Prerequisite Reading:**
- *Quantitative Trading* by Ernest P. Chan
- *Algorithmic Trading: Winning Strategies and Their Rationale* by Ernie Chan
- Research on dynamic risk management and position sizing techniques.

---

## Trading (Algorithm) Hypotheses

The core hypothesis of this strategy is that combining technical indicators (ATR, RSI, SMA) and momentum signal in VN30F1M data and VN30 data (since VN30 and VN30F1M is highly correlated) with dynamic position sizing can produce consistent, profitable trading signals. By scaling into positions based on volatility and signal strength, and applying robust risk controls (stop-losses, take-profits, partial exits, and trailing stops), the strategy aims to capture market opportunities while limiting downside risk.

*Step 1 of the Nine-Step: Formulate the hypothesis that adaptive position sizing and risk management yield superior performance compared to fixed sizing methods.*

---

# Data

## Data Source
- Historical market data including OHLC (Open, High, Low, Close) prices, trading volumes (from Algotrade database), and VN30 index values (from SSI fast connect data API).
More detail of SSI API can be found [here](https://guide.ssi.com.vn/ssi-products/)
- Data is sourced from reliable financial providers and stored in CSV format.


## Data Type
- Time series data
- Price and volume information
- Computed technical indicators (SMA, ATR, RSI)

## Data Period
- **In-Sample:** e.g., January 2023 to December 2023
- **Out-of-Sample:** e.g., January 2024 to April 2024

## How to Get the Input Data?
- Input data files are available in the `data` directory of the repository.
- Additional data may be acquired via APIs or direct downloads from financial data providers.

## How to Store the Output Data?
- Backtest results and performance metrics are stored as CSV files, with filenames indicating whether they are from in-sample or out-of-sample tests.

## Data Collection
Data is collected by downloading historical records and, if necessary, using real-time API feeds. Pre-processing scripts ensure data is cleaned, aligned, and formatted correctly for subsequent analysis.

## Get In-Sample and Out-Sample data
Running main.py in the first part of the file

*Step 2 of the Nine-Step: Gather and verify all required data for analysis.*

---

## Data Processing

Raw data is cleaned and pre-processed to calculate essential technical indicators. This step includes:
- Calculating SMA, ATR, and RSI values.
- Merging various data sources (price, volume, and index data).
- Aligning time series data for accurate analysis.

*Step 3 of the Nine-Step: Process the raw data to create inputs for the trading algorithm.*

---

# Implementation

## Brief Implementation Description
The trading strategy is implemented in Python and consists of the following modules:
- **Backtesting Engine:** Simulates trade execution, dynamic position management (including partial exits and trailing stops), and performance metric calculation.
- **Optimization Module:** Uses Optuna for parameter tuning to maximize strategy performance.
- **Result & Metric Analysis:** Provides visualization and quantitative analysis of performance (e.g., cumulative PNL, Sharpe Ratio, Maximum Drawdown, Win Rate, contract counts).

## Environment Setup and Replication Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/backtesting-optimization.git
   cd backtesting-optimization

# In-sample Backtesting

## Description
In-sample backtesting is the process of calibrating and validating the trading strategy using historical data from the training period. This step helps fine-tune the model parameters and assess performance before applying the strategy to new data.  
*Step 4 of the Nine-Step*

## Parameters
- Strategy parameters (e.g., SMA window length, momentum lookback, acceleration thresholds, etc.)
- Risk management settings (e.g., stop-loss, take-profit thresholds)
- Initial asset value (e.g., 10,000)

## Data
- **Input Data:** `data/insample.csv`
- **Data Period:** For example, January 2023 to December 2023

## In-sample Backtesting Result
The in-sample backtesting results are summarized in a performance table and visualized with a cumulative PNL chart.  

To see the results, run this command with the ```main.py``` file
```
python main.py
```
---

# Optimization

The optimization step involves fine-tuning the trading strategy parameters to maximize performance, measured by cumulative PNL. This step employs advanced hyperparameter tuning techniques to systematically search the parameter space.  
*Step 5 of the Nine-Step*

## Optimization Process / Methods / Library
- **Library:** [Optuna](https://optuna.org/)
- **Method:** Tree-structured Parzen Estimator (TPE) sampler
- **Process:** Multiple trials are conducted to evaluate various parameter combinations, selecting the best based on the objective function (maximizing cumulative PNL).

## Parameters to Optimize
- SMA window length  
- SMA gap  
- Momentum lookback  
- Acceleration thresholds (long and short)  
- Take-profit threshold  
- Cut-loss threshold  
- Quantity window  
- Quantity multiplier  
- Short extra profit  
- RSI window  
- RSI threshold

## Hyper-parameters of the Optimization Process
- **Number of Trials:** 10000 
- **Sampler Seed:** 42  
- **Study Direction:** Maximize cumulative PNL

## Optimization Result
The best parameter set is presented in a summary table with corresponding performance metrics.  

The results is stored in optimization/sma.db
The best parameters is stored in ```optimization/best_params.json``` file
```
{
    "sma_window_length": 25,
    "sma_gap": 0.02017081163304353,
    "momentum_lookback": 7,
    "acceleration_threshold": 0.27265410618834895,
    "short_acceleration_threshold": 0.08577163230799632,
    "take_profit_threshold": 4.9983573167104245,
    "cut_loss_threshold": 1.955171845695386,
    "quantity_window": 20,
    "quantity_multiply": 2,
    "short_extra_profit": 1.9352589805341556,
    "rsi_window": 38,
    "rsi_threshold": 39
}
```
---

# Out-of-sample Backtesting

## Description
Out-of-sample backtesting tests the robustness of the trading strategy using data that was not part of the optimization process. This evaluation determines how well the strategy generalizes to new market conditions.  
*Step 6 of the Nine-Step*

## Parameters
- The optimized strategy parameters (from the optimization step)
- Initial asset value (e.g., 15,000)

## Data
- **Input Data:** `data/outsample.csv`
- **Data Period:** For example, January 2024 to April 2024

## Out-of-sample Backtesting Result
The results include key performance metrics such as cumulative PNL, Sharpe Ratio, Maximum Drawdown, and win rate. These are summarized in tables and visualized in charts.  

To see the results, run this command with the ```main.py``` file
```
python main.py
```
# Conclusion

In conclusion, this project demonstrates that a systematic trading strategy—when coupled with dynamic position sizing and rigorous risk management—can adapt effectively to varying market conditions. The backtesting and optimization process confirms that the strategy not only performs well in historical (in-sample) tests but also generalizes effectively to out-of-sample data, highlighting its potential for practical trading applications.

---

# References

- Chan, E. P. (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*. Wiley.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Relevant research articles and industry reports on dynamic position sizing, risk management, and algorithmic trading methodologies.

---