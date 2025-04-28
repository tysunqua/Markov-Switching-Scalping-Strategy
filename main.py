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

# In-sample backtest
# insample_result = result.backtest_insample_data()
# print("Insample Backtest Result:")
# print(insample_result.head())
# metrics = Metric(insample_result)
# metrics.show_metrics()
# metrics.plot_pnl()

# # Out-of-sample backtest
# outsample_result = result.backtest_outsample_data()
# print("Out-of-sample Backtest Result:")
# print(outsample_result.head())
# metrics = Metric(outsample_result)
# metrics.show_metrics()
# metrics.plot_pnl()

# Reversion strategy backtest
# Insample reversion strategy backtest
reversion_insample_result = result.backtest_reversion_strategy_insample_data()
print("Reversion Insample Backtest Result:")
print(reversion_insample_result.head())
metrics = Metric(reversion_insample_result)
metrics.show_metrics()
metrics.plot_pnl()

# Out-of-sample reversion strategy backtest
reversion_outsample_result = result.backtest_reversion_strategy_outsample_data()
print("Reversion Out-of-sample Backtest Result:")
print(reversion_outsample_result.head())
metrics = Metric(reversion_outsample_result)
metrics.show_metrics()
metrics.plot_pnl()

# Momentum strategy backtest
# Insample momentum strategy backtest
momentum_insample_result = result.backtest_momentum_strategy_insample_data()
print("Momentum Insample Backtest Result:")
print(momentum_insample_result.head())
metrics = Metric(momentum_insample_result)
metrics.show_metrics()
metrics.plot_pnl()

# Out-of-sample momentum strategy backtest
momentum_outsample_result = result.backtest_momentum_strategy_outsample_data()
print("Momentum Out-of-sample Backtest Result:")
print(momentum_outsample_result.head())
metrics = Metric(momentum_outsample_result)
metrics.show_metrics()
metrics.plot_pnl()

# Final Markov strategy backtest
# Insample final Markov strategy backtest
final_markov_insample_result = result.backtest_final_markov_strategy_insample_data()
print("Final Markov Insample Backtest Result:")
print(final_markov_insample_result.head())
metrics = Metric(final_markov_insample_result)
metrics.show_metrics()
metrics.plot_pnl()

# Out-of-sample final Markov strategy backtest
final_markov_outsample_result = result.backtest_final_markov_strategy_outsample_data()
print("Final Markov Out-of-sample Backtest Result:")
print(final_markov_outsample_result.head())
metrics = Metric(final_markov_outsample_result)
metrics.show_metrics()
metrics.plot_pnl()



