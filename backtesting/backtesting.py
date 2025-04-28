from tqdm import tqdm
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import ta

class Backtesting:
    # Global parameters as class attributes
    MAX_TOTAL_CONTRACTS = 45      # Global cap across all positions
    ATR_BASELINE = 1.0            # Baseline ATR value; adjust based on instrument
    TRADING_FEE = 0.47            # Trading fee per contract (example)
    TRAIL_MULTIPLIER = 1.5        # Multiplier to compute trailing stop distance

    def __init__(self):
        # (Optional) Place to initialize instance-specific parameters if needed.
        pass

    # -------------------------------
    # Helper Indicator Functions
    # -------------------------------
    def ATR(self, data, window=14):
        """
        Compute the Average True Range (ATR) of the instrument.
        Assumes data has 'high', 'low', and 'close' columns.
        """
        data = data.copy()
        data['H-L'] = data['high'] - data['low']
        data['H-PC'] = abs(data['high'] - data['close'].shift(1))
        data['L-PC'] = abs(data['low'] - data['close'].shift(1))
        data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        data['ATR'] = data['TR'].rolling(window=window).mean()
        return data

    def RSI(self, data, window=14):
        """
        Compute the Relative Strength Index.
        """
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def momentum_indicators(self, data, params):
        """
        Compute momentum indicators for high volatility regime.
        Assumes data has 'high', 'low', 'close', and 'volume' columns.
        """
        data = data.copy()
        # Getting parameters from params dictionary
        momentum_lookback = params.get("momentum_lookback")
        acceleration_threshold = params.get("acceleration_threshold")
        short_acceleration_threshold = params.get("short_acceleration_threshold")
        ema_fast_period = params.get("ema_fast_period")
        ema_slow_period = params.get("ema_slow_period")
        adx_window = params.get("adx_window")
        macd_window = params.get("macd_window")
        bb_high_window = params.get("bb_high_window")
        volume_window = params.get("quantity_window")
        
        data['Acceleration'] = data['close'] - data['close'].shift(momentum_lookback)
        data['EMA_fast'] = data['close'].ewm(span=ema_fast_period, adjust=False).mean()
        data['EMA_slow'] = data['close'].ewm(span=ema_slow_period, adjust=False).mean()
        data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=adx_window)
        data['ADX_plus_DI'] = ta.trend.adx_pos(data['high'], data['low'], data['close'], window=adx_window)
        data['ADX_minus_DI'] = ta.trend.adx_neg(data['high'], data['low'], data['close'], window=adx_window)
        macd = ta.trend.MACD(data['close'])
        data['MACD'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(data['close'])
        data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        data['Volume_MA'] = data['volume'].rolling(window=volume_window).mean()
        
        data.dropna(inplace=True)
        return data

    def reversion_indicators(self, data, params):
        """
        Compute mean reversion indicators for low volatility regime.
        Assumes data has 'high', 'low', 'close', and 'volume' columns.
        """
        data = data.copy()
        # Getting parameters from params dictionary
        sma_window_length = params.get("sma_window_length")
        rsi_window = params.get("rsi_window")
        bb_window = params.get("bb_window")
        williams_r_window = params.get("williams_r_window")
        cci_window = params.get("cci_window")
        volume_window = params.get("quantity_window")
        stochastic_oscillator_window = params.get("stochastic_oscillator_window")
        
        
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'], window=stochastic_oscillator_window)
        data['Stoch_%K'] = stoch.stoch()
        bb = ta.volatility.BollingerBands(data['close'], window=bb_window)
        data['BB_low'] = bb.bollinger_lband()
        data['BB_high'] = bb.bollinger_hband()
        data['Williams_%R'] = ta.momentum.williams_r(data['high'], data['low'], data['close'], lbp=williams_r_window, fillna=True)
        data['CCI'] = ta.trend.cci(data['high'], data['low'], data['close'], window=cci_window)
        data['SMA'] = data['close'].rolling(window=sma_window_length).mean()
        data['Price/SMA'] = data['close'] / data['SMA']
        data['RSI'] = self.RSI(data, rsi_window)
        
        data.dropna(inplace=True)
        return data
    
    # HMM Volatility Clustering
    def detect_volatility_regimes(self, data):
        data = data.copy()
        # Create ATR for 1-minute data
        data = self.ATR(data, window=14)
        #Convert 1-minute data to 5-minute data
        data_5min = data.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        returns = data_5min['close'].pct_change().dropna()
        data_5min = self.ATR(data_5min, window=14)
        atr = data_5min['ATR'].values[-len(returns):]
        atr = atr[~np.isnan(atr)]  # Remove NaN values from ATR
        features = np.column_stack([returns[-len(atr):], atr])

        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)
        model.fit(features)
        regimes = model.predict(features)

        data_5min = data_5min.iloc[-len(regimes):].copy()
        data_5min['Regime'] = regimes
        
        #Map 5-min regimes back to 1-min candles
        data['Regime'] = data_5min['Regime']
        data['Regime'] = data['Regime'].ffill()

        # Identify which state is high/low volatility
        vol_by_regime = data.groupby('Regime')['ATR'].mean()
        high_vol_state = vol_by_regime.idxmax()
        data['Volatility'] = data['Regime'].apply(lambda x: 'High' if x == high_vol_state else 'Low')

        return data
    
    def predict_volatility_regimes_test(self, train, test):
        train = train.copy()
        test = test.copy()

        # Compute ATR on training data
        train = self.ATR(train, window=14)

        # Convert training data to 5-minute candles
        train_5min = train.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Prepare features (returns and ATR) for training data
        returns_train = train_5min['close'].pct_change().dropna()
        train_5min = self.ATR(train_5min, window=14)
        atr_train = train_5min['ATR'].values[-len(returns_train):]
        atr_train = atr_train[~np.isnan(atr_train)]
        features_train = np.column_stack([returns_train[-len(atr_train):], atr_train])

        # Fit the Gaussian HMM model using the training data
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10000)
        model.fit(features_train)

        # Now prepare test data
        test = self.ATR(test, window=14)
        test_5min = test.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        returns_test = test_5min['close'].pct_change().dropna()
        test_5min = self.ATR(test_5min, window=14)
        atr_test = test_5min['ATR'].values[-len(returns_test):]
        atr_test = atr_test[~np.isnan(atr_test)]
        features_test = np.column_stack([returns_test[-len(atr_test):], atr_test])

        # Predict regimes on test data
        regimes = model.predict(features_test)

        test_5min = test_5min.iloc[-len(regimes):].copy()
        test_5min['Regime'] = regimes

        # Map 5-minute regimes back to 1-minute test data
        test['Regime'] = test_5min['Regime']
        test['Regime'] = test['Regime'].ffill()

        # Identify high/low volatility regimes from test set for consistency
        vol_by_regime = test.groupby('Regime')['ATR'].mean()
        high_vol_state = vol_by_regime.idxmax()
        test['Volatility'] = test['Regime'].apply(lambda x: 'High' if x == high_vol_state else 'Low')

        return test


    # -------------------------------
    # Dynamic Sizing Functions
    # -------------------------------

    # Adjusted signal strength calculation for two regimes
    def calculate_signal_strength(self, row, regime, params):
        if regime == 'High':  # Momentum strategy in high volatility
            ema_fast = row['EMA_fast']
            ema_slow = row['EMA_slow']
            if ema_fast > ema_slow and row['Acceleration'] > params['acceleration_threshold']:
                return min(row['Acceleration'] / params['acceleration_threshold'], 1)
            elif ema_fast < ema_slow and row['Acceleration'] < -params['acceleration_threshold']:
                return min(abs(row['Acceleration']) / params['acceleration_threshold'], 1)
            return 0
        else:  # Reversion strategy in low volatility
            price_sma_gap = abs(row['Price/SMA'] - 1)
            rsi_diff = abs(row['RSI'] - 50)
            if price_sma_gap >= params['sma_gap'] and rsi_diff >= params['rsi_threshold']:
                return min((price_sma_gap / params['sma_gap'] + rsi_diff / params['rsi_threshold']) / 2, 1)
            return 0

    # Adjusted position condition checks for two regimes
    def check_position_conditions(self, row, prev_row, regime, params, direction):
        if regime == 'High':  # Momentum strategy conditions
            ema_fast = row['EMA_fast']
            ema_slow = row['EMA_slow']
            prev_ema_fast = prev_row['EMA_fast']
            prev_ema_slow = prev_row['EMA_slow']
            if direction == 'LONG':
                conditions = [
                    ((ema_fast > ema_slow) and (prev_ema_fast < prev_ema_slow)),
                    row['Acceleration'] > params['acceleration_threshold'],
                    (row['ADX'] > params['adx_threshold'] and row['ADX_plus_DI'] > row['ADX_minus_DI']),
                    (row['MACD'] > 0 and row['MACD'] > prev_row['MACD']),
                    (row['close'] > row['BB_high']),
                    (row['OBV'] > prev_row['OBV']),
                    (row['volume'] > 1.5 * row['Volume_MA'])
                ]
            else:  # SHORT
                conditions = [
                    ((ema_fast < ema_slow) and (prev_ema_fast > prev_ema_slow)),
                    row['Acceleration'] < -params['acceleration_threshold'],
                    (row['ADX'] > params['adx_threshold'] and row['ADX_minus_DI'] > row['ADX_plus_DI']),
                    (row['MACD'] < 0 and row['MACD'] < prev_row['MACD']),
                    (row['close'] < row['BB_low']),
                    (row['OBV'] < prev_row['OBV']),
                    (row['volume'] > 1.5 * row['Volume_MA'])
                ]
        else:  # Reversion strategy conditions in low volatility
            if direction == 'LONG':
                conditions = [
                    row['Price/SMA'] < 1 - params['sma_gap'],
                    row['RSI'] < 50 - params['rsi_threshold'],
                    (row['Stoch_%K'] < 50 - params['stochastic_oscillator_threshold']),
                    (row['close'] < row['BB_low']),
                    (row['Williams_%R'] < -50 - params['willr_threshold']),
                    (row['CCI'] < -params['cci_threshold']),
                    (row['volume'] > 2 * row['Volume_MA'])
                ]
            else:  # SHORT
                conditions = [
                    row['Price/SMA'] > 1 + params['sma_gap'],
                    row['RSI'] > 50 + params['rsi_threshold'],
                    (row['Stoch_%K'] > 50 + params['stochastic_oscillator_threshold']),
                    (row['close'] > row['BB_high']),
                    (row['Williams_%R'] > -50 + params['willr_threshold']),
                    (row['CCI'] > params['cci_threshold']),
                    (row['volume'] > 2 * row['Volume_MA'])
                ]
        conditions_met = sum(conditions)
        return conditions_met >= len(conditions) - 1  # Allow at most two conditions to fail

    # Adjusted exit thresholds for two regimes
    def get_thresholds(self, regime, params):
        if regime == 'High':
            return params['take_profit_threshold_high'], params['cut_loss_threshold_high']
        else:
            return params['take_profit_threshold_low'], params['cut_loss_threshold_low'] # Adjusted method for calculating contracts based on volatility regimes
    def calculate_contracts(self, regime, signal_strength):
        if regime == 'High':
            adjusted = max(5, int(round(signal_strength * 5)))  # Smaller size in high volatility
        else:  # Low volatility
            adjusted = max(10, int(round(signal_strength * 12)))  # Larger size in low volatility
        return adjusted


    def get_allowed_size(self, desired_size, current_total):
        """
        Limit the size by the remaining capacity (global cap 45).
        """
        available = self.MAX_TOTAL_CONTRACTS - current_total
        return max(0, min(desired_size, available))

    # -------------------------------
    # Position Management Functions
    # -------------------------------
    def open_position(self, position_type, entry_point, contracts, holdings):
        """
        Create a new position represented as a dictionary.
        """
        position = {
            'position_type': position_type,
            'entry_price': entry_point,
            'contracts': contracts,        # current number of contracts in this position
            'has_partial_exited': False,   # flag to indicate partial exit occurred
            'trailing_stop': None          # will be set once partial exit is taken
        }
        holdings.append(position)
        return holdings

    def partial_close_position(self, position, cur_price, partial_fraction=0.5):
        """
        Exit a portion of the position.
        Returns realized PnL and number of contracts closed.
        """
        closed_contracts = int(round(position['contracts'] * partial_fraction))
        if closed_contracts < 1:
            closed_contracts = 1
        if position['position_type'] == 'LONG':
            realized_pnl = (cur_price - position['entry_price']) * closed_contracts - self.TRADING_FEE * closed_contracts
        else:  # SHORT
            realized_pnl = (position['entry_price'] - cur_price) * closed_contracts - self.TRADING_FEE * closed_contracts
        position['contracts'] -= closed_contracts
        position['has_partial_exited'] = True
        if position['position_type'] == 'LONG':
            position['trailing_stop'] = position['entry_price'] + self.TRADING_FEE
        else:
            position['trailing_stop'] = position['entry_price'] - self.TRADING_FEE
        return realized_pnl, closed_contracts

    def close_full_position(self, position, cur_price):
        """
        Fully exit the position.
        """
        contracts = position['contracts']
        if position['position_type'] == 'LONG':
            realized_pnl = (cur_price - position['entry_price']) * contracts - self.TRADING_FEE * contracts
        else:
            realized_pnl = (position['entry_price'] - cur_price) * contracts - self.TRADING_FEE * contracts
        return realized_pnl, contracts

    def update_trailing_stop(self, position, cur_price, trail_distance):
        """
        For LONG: move stop upward if (cur_price - trail_distance) exceeds current trailing stop.
        For SHORT: move stop downward if (cur_price + trail_distance) is lower than current trailing stop.
        """
        if position['position_type'] == 'LONG':
            new_stop = cur_price - trail_distance
            if new_stop > position['trailing_stop']:
                position['trailing_stop'] = new_stop
        else:
            new_stop = cur_price + trail_distance
            if new_stop < position['trailing_stop']:
                position['trailing_stop'] = new_stop
        return position

    # -------------------------------
    # Condition Functions
    # -------------------------------
    # def check_long_position_conditions(self, row, acceleration_threshold, quantity_multiply, sma_gap, short_acceleration_threshold, rsi_threshold):
    #     conditions = [
    #         row['Acceleration'] > acceleration_threshold,
    #         row['VN30 Acceleration'] > 0,
    #         row['volume'] > row['Average Quantity'] * quantity_multiply,
    #         row['Price/SMA'] < 1 - sma_gap,
    #         row['Short Acceleration'] > short_acceleration_threshold,
    #         row['RSI'] < 50 - rsi_threshold
    #     ]
    #     # Allow at most one condition to fail
    #     return conditions.count(True) >= len(conditions) - 2

    # def check_short_position_conditions(self, row, acceleration_threshold, quantity_multiply, sma_gap, short_acceleration_threshold, rsi_threshold):
    #     conditions = [
    #         row['Acceleration'] < -acceleration_threshold,
    #         row['VN30 Acceleration'] < 0,
    #         row['volume'] > row['Average Quantity'] * quantity_multiply,
    #         row['Price/SMA'] > 1 + sma_gap,
    #         row['Short Acceleration'] < -short_acceleration_threshold,
    #         row['RSI'] > 50 + rsi_threshold
    #     ]
    #     # Allow at most two conditions to fail
    #     return conditions.count(True) >= len(conditions) - 2

    # -------------------------------
    # Main Backtesting Function
    # -------------------------------
    def run(self, trading_data, params, asset_value=10000):
        """
        Run the backtesting strategy using the provided trading data and parameter dictionary.
        
        The params dictionary should contain:
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
        """
        # -------------------------------
        # Preprocess data and compute indicators
        # -------------------------------
        # trading_data = trading_data.copy()
        # sma_window_length = params.get("sma_window_length")
        # ema_fast_period = params.get("ema_fast_period")
        # ema_slow_period = params.get("ema_slow_period")
        # momentum_lookback = params.get("momentum_lookback")
        # quantity_window = params.get("quantity_window")
        
        # trading_data['SMA'] = trading_data['close'].rolling(sma_window_length).mean()
        # trading_data['EMA_fast'] = trading_data['close'].ewm(span=ema_fast_period, adjust=False).mean()
        # trading_data['EMA_slow'] = trading_data['close'].ewm(span=ema_slow_period, adjust=False).mean()
        # trading_data['Price/SMA'] = trading_data['close'] / trading_data['SMA']
        # trading_data['Average Quantity'] = trading_data['volume'].rolling(quantity_window).mean()
        # trading_data['Acceleration'] = trading_data['close'] - trading_data['close'].shift(momentum_lookback)
        # trading_data['RSI'] = self.RSI(trading_data, params.get("rsi_window"))
        trading_data = self.momentum_indicators(trading_data, params)
        trading_data = self.reversion_indicators(trading_data, params)
        trading_data = self.ATR(trading_data, window=14)
        trading_data.dropna(inplace=True)
        
        
        # Map parameter keys to the ones expected by get_thresholds.
        params['take_profit_threshold_high'] = params.get("take_profit_threshold_high_volatility")
        params['take_profit_threshold_low'] = params.get("take_profit_threshold_low_volatility")
        params['cut_loss_threshold_high'] = params.get("cut_loss_threshold_high_volatility")
        params['cut_loss_threshold_low'] = params.get("cut_loss_threshold_low_volatility")
        
        # -------------------------------
        # Initialize bookkeeping variables
        # -------------------------------
        trading_data['Contracts Held'] = 0
        trading_data['Cumulative Long'] = 0
        trading_data['Cumulative Short'] = 0
        trading_data['Asset'] = asset_value
        trading_data['PNL'] = 0
        trading_data['Cumulative PNL'] = 0
        trading_data['Position'] = None
        trading_data['Entry Price'] = None
        
        holdings = []              # list of open positions
        total_open_contracts = 0    # global count of contracts currently held
        cumulative_pnl = 0
        
        cumulative_long_contracts = 0
        cumulative_short_contracts = 0
        
        asset_history = []
        pnl_history = []
        cumulative_pnl_history = []
        
        # -------------------------------
        # Process each trading bar
        # -------------------------------
        for i in tqdm(range(len(trading_data))):
            total_realized_pnl = 0
            cur_price = trading_data['close'].iloc[i]
            row = trading_data.iloc[i]
            current_atr = row['ATR']
            regime = row['Volatility']  # Should be "High" or "Low"
            
            if i == 0:
                prev_row = row
            if i != 0:
                prev_row = trading_data.iloc[i - 1]
            
            # Obtain thresholds for the current regime.
            tp_threshold, sl_threshold = self.get_thresholds(regime, params)
            
            # -------------------------------
            # EXIT STRATEGY
            # -------------------------------
            for pos in holdings[:]:
                if pos['position_type'] == 'LONG':
                    # Stop loss exit
                    if cur_price < pos['entry_price'] - sl_threshold:
                        pnl, closed = self.close_full_position(pos, cur_price)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        holdings.remove(pos)
                        continue
                    # Partial exit for profit taking.
                    if cur_price >= pos['entry_price'] + tp_threshold and not pos['has_partial_exited']:
                        pnl, closed = self.partial_close_position(pos, cur_price, partial_fraction=0.5)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        pos['trailing_stop'] = pos['entry_price'] + tp_threshold
                    # Exit if price falls below trailing stop.
                    if pos['has_partial_exited'] and pos['trailing_stop'] is not None and cur_price < pos['trailing_stop']:
                        pnl, closed = self.close_full_position(pos, cur_price)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        holdings.remove(pos)
                        continue
                    # Update trailing stop using the current ATR.
                    if pos['has_partial_exited'] and pos['trailing_stop'] is not None:
                        trail_distance = self.TRAIL_MULTIPLIER * current_atr
                        pos = self.update_trailing_stop(pos, cur_price, trail_distance)
                        
                elif pos['position_type'] == 'SHORT':
                    # Stop loss exit for shorts.
                    if cur_price > pos['entry_price'] + sl_threshold:
                        pnl, closed = self.close_full_position(pos, cur_price)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        holdings.remove(pos)
                        continue
                    # Partial exit for shorts (optionally using short_extra_profit).
                    if cur_price <= pos['entry_price'] - (tp_threshold + params.get("short_extra_profit", 0)) and not pos['has_partial_exited']:
                        pnl, closed = self.partial_close_position(pos, cur_price, partial_fraction=0.5)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        pos['trailing_stop'] = pos['entry_price'] - tp_threshold
                    # Exit if price moves against the trailing stop.
                    if pos['has_partial_exited'] and pos['trailing_stop'] is not None and cur_price > pos['trailing_stop']:
                        pnl, closed = self.close_full_position(pos, cur_price)
                        total_realized_pnl += pnl
                        total_open_contracts -= closed
                        holdings.remove(pos)
                        continue
                    # Update trailing stop.
                    if pos['has_partial_exited'] and pos['trailing_stop'] is not None:
                        trail_distance = self.TRAIL_MULTIPLIER * current_atr
                        pos = self.update_trailing_stop(pos, cur_price, trail_distance)
            
            # Update asset value and cumulative pnl
            asset_value += total_realized_pnl
            cumulative_pnl += total_realized_pnl
            asset_history.append(asset_value)
            pnl_history.append(total_realized_pnl)
            cumulative_pnl_history.append(cumulative_pnl)
            
            # -------------------------------
            # ENTRY STRATEGY
            # -------------------------------
            # LONG entry (only if no short position exists)
            if not holdings or (holdings and holdings[0]['position_type'] != 'SHORT'):
                if self.check_position_conditions(row, prev_row, regime, params, 'LONG'):
                    signal_strength = self.calculate_signal_strength(row, regime, params)
                    desired_contracts = self.calculate_contracts(regime, signal_strength)
                    existing_long = None
                    for pos in holdings:
                        if pos['position_type'] == 'LONG':
                            existing_long = pos
                            break
                    if existing_long:
                        additional_desired = desired_contracts
                        allowed_additional = self.get_allowed_size(additional_desired, total_open_contracts)
                        if allowed_additional > 0:
                            total_contracts_before = existing_long['contracts']
                            total_contracts_after = total_contracts_before + allowed_additional
                            existing_long['entry_price'] = (
                                existing_long['entry_price'] * total_contracts_before + cur_price * allowed_additional
                            ) / total_contracts_after
                            existing_long['contracts'] = total_contracts_after
                            total_open_contracts += allowed_additional
                            cumulative_long_contracts += allowed_additional
                    else:
                        allowed = self.get_allowed_size(desired_contracts, total_open_contracts)
                        if allowed > 0:
                            holdings = self.open_position('LONG', cur_price, allowed, holdings)
                            total_open_contracts += allowed
                            cumulative_long_contracts += allowed
            
            # SHORT entry (only if no long position exists)
            if not holdings or (holdings and holdings[0]['position_type'] != 'LONG'):
                if self.check_position_conditions(row, prev_row, regime, params, 'SHORT'):
                    signal_strength = self.calculate_signal_strength(row, regime, params)
                    desired_contracts = self.calculate_contracts(regime, signal_strength)
                    existing_short = None
                    for pos in holdings:
                        if pos['position_type'] == 'SHORT':
                            existing_short = pos
                            break
                    if existing_short:
                        additional_desired = desired_contracts
                        allowed_additional = self.get_allowed_size(additional_desired, total_open_contracts)
                        if allowed_additional > 0:
                            total_contracts_before = existing_short['contracts']
                            total_contracts_after = total_contracts_before + allowed_additional
                            existing_short['entry_price'] = (
                                existing_short['entry_price'] * total_contracts_before + cur_price * allowed_additional
                            ) / total_contracts_after
                            existing_short['contracts'] = total_contracts_after
                            total_open_contracts += allowed_additional
                            cumulative_short_contracts += allowed_additional
                    else:
                        allowed = self.get_allowed_size(desired_contracts, total_open_contracts)
                        if allowed > 0:
                            holdings = self.open_position('SHORT', cur_price, allowed, holdings)
                            total_open_contracts += allowed
                            cumulative_short_contracts += allowed
            
            # Record position and counts for this bar.
            if holdings:
                trading_data.at[trading_data.index[i], 'Position'] = holdings[0]['position_type']
                trading_data.at[trading_data.index[i], 'Entry Price'] = holdings[0]['entry_price']
            else:
                trading_data.at[trading_data.index[i], 'Position'] = None
                trading_data.at[trading_data.index[i], 'Entry Price'] = None
            
            trading_data.at[trading_data.index[i], 'Contracts Held'] = total_open_contracts
            trading_data.at[trading_data.index[i], 'Cumulative Long'] = cumulative_long_contracts
            trading_data.at[trading_data.index[i], 'Cumulative Short'] = cumulative_short_contracts
        
        # Save history
        trading_data['Asset'] = asset_history
        trading_data['PNL'] = pnl_history
        trading_data['Cumulative PNL'] = cumulative_pnl_history
        
        return trading_data

# Create an instance of Backtesting
backtesting = Backtesting()
