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


if __name__ == "__main__":
    choice = input("Select backtest type ('in', 'out', or 'both'): ").strip().lower()
    if choice == "in":
        momentum_insample_result    = result.backtest_momentum_strategy_insample_data()
        final_markov_insample_result  = result.backtest_final_markov_strategy_insample_data()
        metrics_insample = Metric(momentum_insample_result, final_markov_insample_result)
        metrics_insample.plot_pnl2(title1="Momentum Strategy Insample", title2="Markov Strategy Insample")
        metrics_insample.show_metrics()

    elif choice == "out":
        momentum_outsample_result   = result.backtest_momentum_strategy_outsample_data()
        final_markov_outsample_result = result.backtest_final_markov_strategy_outsample_data()
        metrics_outsample = Metric(momentum_outsample_result, final_markov_outsample_result)
        metrics_outsample.plot_pnl2(title1="Momentum Strategy Outsample", title2="Markov Strategy Outsample")
        metrics_outsample.show_metrics()

    elif choice == "both":
        momentum_insample_result    = result.backtest_momentum_strategy_insample_data()
        momentum_outsample_result   = result.backtest_momentum_strategy_outsample_data()
        final_markov_insample_result  = result.backtest_final_markov_strategy_insample_data()
        final_markov_outsample_result = result.backtest_final_markov_strategy_outsample_data()

        metrics_insample = Metric(momentum_insample_result, final_markov_insample_result)
        metrics_outsample = Metric(momentum_outsample_result, final_markov_outsample_result)

        metrics_insample.plot_pnl2(title1="Momentum Strategy Insample", title2="Markov Strategy Insample")
        metrics_outsample.plot_pnl2(title1="Momentum Strategy Outsample", title2="Markov Strategy Outsample")

        metrics_insample.show_metrics()
        metrics_outsample.show_metrics()
    else:
        print("Invalid choice. Please select 'insample', 'outsample', or 'both'.")


