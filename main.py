from data.service import *
from backtesting.backtesting import *
from config.config import optimization_params
from optimization.optimization import *
from performance.result import BacktestResult
from performance.metric import Metric
from backtesting.backtesting import Backtesting
import pandas as pd

# Load the data
data_service = DataService()
train_data = data_service.get_train_data()
test_data = data_service.get_test_data()

# Print the head of the data
print("Train Data:")
print(train_data.head())
print("Test Data:")
print(test_data.head())

# # Initialize the backtesting object
backtesting = Backtesting()
# Testing hidden markov model
hmm_data = backtesting.detect_volatility_regimes(train_data)

print("Hidden Markov Model Data:")
print(hmm_data.head())



# Run the backtest with the best parameters
result = BacktestResult(optimization_params)

import matplotlib.pyplot as plt

# Run your backtests
momentum_insample_result    = result.backtest_momentum_strategy_insample_data()
momentum_outsample_result   = result.backtest_momentum_strategy_outsample_data()
final_markov_insample_result  = result.backtest_final_markov_strategy_insample_data()
final_markov_outsample_result = result.backtest_final_markov_strategy_outsample_data()

# Get the performance metrics
momentum_insample_metrics = Metric(momentum_insample_result)
momentum_outsample_metrics = Metric(momentum_outsample_result)
final_markov_insample_metrics = Metric(final_markov_insample_result)
final_markov_outsample_metrics = Metric(final_markov_outsample_result)

# Prepare the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 3) Momentum Insample
metrics = Metric(momentum_insample_result)
plt.sca(axes[0, 0])                # set current axis
metrics.plot_pnl()                 # draws here
axes[0, 0].set_title('Momentum Insample Backtest')

# 4) Momentum Outsample
metrics = Metric(momentum_outsample_result)
plt.sca(axes[0, 1])
metrics.plot_pnl()
axes[0, 1].set_title('Momentum Outsample Backtest')

# 5) Final Markov Insample
metrics = Metric(final_markov_insample_result)
plt.sca(axes[1, 0])
metrics.plot_pnl()
axes[1, 0].set_title('Final Markov Insample Backtest')

# 6) Final Markov Outsample
metrics = Metric(final_markov_outsample_result)
plt.sca(axes[1, 1])
metrics.plot_pnl()
axes[1, 1].set_title('Final Markov Outsample Backtest')

# 7) Tidy up
plt.tight_layout()
plt.show()




