from data.service import *
from backtesting.backtesting import *
from config.config import optimization_params
from optimization.optimization import *
from performance.result import BacktestResult
from performance.metric import Metric
from backtesting.backtesting import Backtesting
import pandas as pd

data_service = DataService()
# train = data_service.get_matched_data("2020-01-01", "2023-07-01")
# test = data_service.get_matched_data("2023-07-01", "2025-05-01")

# # Save the data to CSV files
# train.to_csv("data/train.csv", index=True)
# test.to_csv("data/test.csv", index=True)

train = data_service.get_vn30f_data("2020-01-01", "2023-12-31")
# test = data_service.get_vn30f_data("2024-01-01", "2025-04-30")

# Save the data to CSV files
train.to_csv("data/train.csv", index=True)
# test.to_csv("data/test.csv", index=True)