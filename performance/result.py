import pandas as pd
from backtesting.backtesting import Backtesting  # Adjust this import based on your project structure
from backtesting.momentum_backtesting import MomentumBacktesting  # Adjust this import based on your project structure
from backtesting.final_markov_backtesting import FinalMarkovBacktesting  # Adjust this import based on your project structure
from backtesting.reversion_backtesting import ReversionBacktesting  # Adjust this import based on your project structure

class BacktestResult:
    def __init__(self, params, asset_value=15000):
        """
        Initialize with the parameters for the backtest and an initial asset value.
        
        Parameters:
            params (dict): A dictionary of strategy parameters.
            asset_value (float): Starting asset value (default: 10000).
        """
        self.params = params
        self.asset_value = asset_value
        self.backtester = Backtesting()
        self.reversion_backtester = ReversionBacktesting()
        self.momentum_backtester = MomentumBacktesting()
        self.final_markov_backtester = FinalMarkovBacktesting()

    def backtest_insample_data(self, file_path = "data/train.csv"):
        """
        Run the backtesting strategy on insample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing insample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        insample_data = pd.read_csv(file_path)
        # set the datetime column as index
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        insample_data = self.backtester.detect_volatility_regimes(insample_data)
        result = self.backtester.run(insample_data, self.params, self.asset_value)
        return result

    def backtest_outsample_data(self, file_path = "data/test.csv", insample_data_path = "data/train.csv"):
        """
        Run the backtesting strategy on outsample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing outsample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        outsample_data = pd.read_csv(file_path)
        insample_data = pd.read_csv(insample_data_path)
        # set the datetime column as index
        outsample_data.index = pd.to_datetime(outsample_data['datetime'])
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        outsample_data = self.backtester.predict_volatility_regimes_test(outsample_data, outsample_data)
        result = self.backtester.run(outsample_data, self.params, self.asset_value)
        return result
    
    def backtest_momentum_strategy_insample_data(self, file_path = "data/train.csv"):
        """
        Run the momentum strategy backtesting on insample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing insample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        insample_data = pd.read_csv(file_path)
        # set the datetime column as index
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        insample_data = self.momentum_backtester.detect_volatility_regimes(insample_data)
        result = self.momentum_backtester.run(insample_data, self.params, self.asset_value)
        return result

    def backtest_momentum_strategy_outsample_data(self, file_path = "data/test.csv", insample_data_path = "data/train.csv"):
        """
        Run the momentum strategy backtesting on outsample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing outsample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        outsample_data = pd.read_csv(file_path)
        insample_data = pd.read_csv(insample_data_path)
        # set the datetime column as index
        outsample_data.index = pd.to_datetime(outsample_data['datetime'])
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        outsample_data = self.momentum_backtester.predict_volatility_regimes_test(outsample_data, outsample_data)
        result = self.momentum_backtester.run(outsample_data, self.params, self.asset_value)
        return result
    
    def backtest_final_markov_strategy_insample_data(self, file_path = "data/train.csv"):
        """
        Run the final Markov strategy backtesting on insample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing insample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        insample_data = pd.read_csv(file_path)
        # set the datetime column as index
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        insample_data = self.final_markov_backtester.detect_volatility_regimes(insample_data)
        result = self.final_markov_backtester.run(insample_data, self.params, self.asset_value)
        return result
    
    def backtest_final_markov_strategy_outsample_data(self, file_path = "data/test.csv", insample_data_path = "data/train.csv"):
        """
        Run the final Markov strategy backtesting on outsample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing outsample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        outsample_data = pd.read_csv(file_path)
        insample_data = pd.read_csv(insample_data_path)
        # set the datetime column as index
        outsample_data.index = pd.to_datetime(outsample_data['datetime'])
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        outsample_data = self.final_markov_backtester.predict_volatility_regimes_test(outsample_data, outsample_data)
        result = self.final_markov_backtester.run(outsample_data, self.params, self.asset_value)
        return result
    
    def backtest_reversion_strategy_insample_data(self, file_path = "data/train.csv"):
        """
        Run the reversion strategy backtesting on insample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing insample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        insample_data = pd.read_csv(file_path)
        # set the datetime column as index
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        insample_data = self.reversion_backtester.detect_volatility_regimes(insample_data)
        result = self.reversion_backtester.run(insample_data, self.params, self.asset_value)
        return result
    
    def backtest_reversion_strategy_outsample_data(self, file_path = "data/test.csv", insample_data_path = "data/train.csv"):
        """
        Run the reversion strategy backtesting on outsample data.
        
        Parameters:
            file_path (str): Path to the CSV file containing outsample data.
        
        Returns:
            DataFrame: The result of the backtest.
        """
        outsample_data = pd.read_csv(file_path)
        insample_data = pd.read_csv(insample_data_path)
        # set the datetime column as index
        outsample_data.index = pd.to_datetime(outsample_data['datetime'])
        insample_data.index = pd.to_datetime(insample_data['datetime'])
        outsample_data = self.reversion_backtester.predict_volatility_regimes_test(outsample_data, outsample_data)
        result = self.reversion_backtester.run(outsample_data, self.params, self.asset_value)
        return result