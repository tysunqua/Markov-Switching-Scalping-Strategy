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

backtesting = Backtesting()
for i in range(100):
    print(i)
    study_name = "markov_v10"
    storage = "sqlite:///hmm_results.db"
    n_trials = 10
    seed = 22
    train_data_path = "data/train.csv"

    optimization = Optimization(train_data_path, study_name, storage, n_trials, seed)
    results = optimization.run_optimization()
    optimization.save_best_params(results)