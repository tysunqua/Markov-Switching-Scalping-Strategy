import optuna
import json
import pandas as pd
from backtesting.backtesting import Backtesting  # adjust import according to your module structure

class Optimization:
    def __init__(self, train_data_path, study_name, storage, n_trials, seed=42):
        """
        Initialize the optimization instance.
        
        Parameters:
            train_data_path (str): Path to the CSV file containing training data.
            study_name (str): The name of the Optuna study.
            storage (str): Storage URL for the study (e.g., 'sqlite:///sma.db').
            n_trials (int): Number of optimization trials.
            seed (int): Seed for the sampler (default 42).
        """
        
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.sampler = optuna.samplers.TPESampler(seed=seed)
        self.backtest = Backtesting()
        self.train = pd.read_csv(train_data_path)
        self.train.dropna(inplace=True)
        self.train['datetime'] = pd.to_datetime(self.train['datetime'])
        self.train.set_index("datetime", inplace=True)
        # Determine volatility regime using Markov HMM
        self.trading_data = self.backtest.detect_volatility_regimes(self.train)
    
    def objective(self, trial):
        """
        Objective function for Optuna that suggests parameter values,
        runs the backtesting strategy, and returns the cumulative PNL.
        """
        params = {
            "sma_window_length": trial.suggest_int('sma_window_length', 10, 1000),
            "sma_gap": trial.suggest_float('sma_gap', 0.0005, 0.1),
            "momentum_lookback": trial.suggest_int('momentum_lookback', 2, 10),
            "acceleration_threshold": trial.suggest_float('acceleration_threshold', 0.0, 1),
            "short_acceleration_threshold": trial.suggest_float('short_acceleration_threshold', 0.00, 0.5),
            "take_profit_threshold_high_volatility": trial.suggest_float('take_profit_threshold_high_volatility', 5.5, 5.5),
            "take_profit_threshold_low_volatility": trial.suggest_float('take_profit_threshold_low_volatility', 4.0, 4.0),
            "cut_loss_threshold_high_volatility": trial.suggest_float('cut_loss_threshold_high_volatility', 2.0, 2.0),
            "cut_loss_threshold_low_volatility": trial.suggest_float('cut_loss_threshold_low_volatility', 1.5, 1.5),
            "quantity_window": trial.suggest_int('quantity_window', 2, 50),
            "quantity_multiply": trial.suggest_int('quantity_multiply', 0, 5),
            "short_extra_profit": trial.suggest_float('short_extra_profit', 0, 2),
            "rsi_window": trial.suggest_int('rsi_window', 5, 100),
            "rsi_threshold": trial.suggest_int('rsi_threshold', 5, 45),
            "ema_fast_period": trial.suggest_int('ema_fast_period', 2, 20),
            "ema_slow_period": trial.suggest_int('ema_slow_period', 10, 100),
            "adx_window": trial.suggest_int('adx_window', 5, 100),
            "adx_threshold": trial.suggest_int('adx_threshold', 10, 75),
            "stochastic_oscillator_window": trial.suggest_int('stochastic_oscillator_window', 5, 100),
            "stochastic_oscillator_threshold": trial.suggest_int('stochastic_oscillator_threshold', 10, 30),
            "bb_window": trial.suggest_int('bb_window', 5, 100),
            "cci_window": trial.suggest_int('cci_window', 5, 100),
            "cci_threshold": trial.suggest_int('cci_threshold', 70, 100),
            "williams_r_window": trial.suggest_int('williams_r_window', 1, 10),
            "willr_threshold": trial.suggest_int('willr_threshold', 10, 40),
        }
        result = self.backtest.run(self.trading_data, params)
        # We assume that the last row contains the final cumulative PNL.
        return result.iloc[-1]["Cumulative PNL"]

    def run_optimization(self):
        """
        Create and run the Optuna study, returning the best parameters.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            sampler=self.sampler,
            direction="maximize"
        )
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params

    def save_best_params(self, best_params, filepath = 'optimization/best_params.json'):
        """
        Save the best parameters to a JSON file.
        
        Parameters:
            best_params (dict): The best parameters found by Optuna.
            filepath (str): Path to save the JSON file (default: 'best_params.json').
        """
        with open(filepath, 'w') as f:
            json.dump(best_params, f, indent=4)
        return best_params
    
        
# Example usage:
# optimizer = Optimization(train_data_path='data/train.csv', study_name='sma-vq2', storage='sqlite:///sma.db', n_trials=3000)
# best_parameters = optimizer.save_best_params('best_params.json')
# print("Best Parameters:", best_parameters)
