"""
Modular Backtesting Framework for Cryptocurrency Trading Strategies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from sklearn.model_selection import ParameterGrid
import io
import sys
import abc
from tqdm.auto import tqdm  # Using tqdm.auto instead of regular tqdm for better handling with outputs

# ===============================================================================
# DATA HANDLING
# ===============================================================================

class DataHandler:
    """
    Handles data loading, cleaning, and resampling operations
    """
    @staticmethod
    def load_and_clean_data(csv_file, rows, granularity='1min'):
        """
        Load the last N data points from CSV file
        Clean the data and convert Unix timestamp to human-readable datetime
        Resample to specified granularity
        
        Parameters:
        - csv_file: path to CSV file
        - rows: number of rows to load from the end of the file
        - granularity: timeframe to resample data to ('1min', '5min', '15min', etc.)
        """
        print(f"Loading data from {csv_file}...")
        
        # Check file exists and get file size
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"File not found: {csv_file}")
        
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # Count total rows to calculate how many to skip
        total_rows = sum(1 for _ in open(csv_file))
        skip_rows = max(0, total_rows - rows)
        print(f"Total rows: {total_rows}, reading last {rows} rows (skipping {skip_rows})")
        
        # Read the CSV file, skipping initial rows to get only the last N
        df = pd.read_csv(csv_file, skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None)
        
        print(f"Data loaded, shape: {df.shape}")
        print("Column names:", df.columns.tolist())
        
        # Check if 'unix' or timestamp column exists
        timestamp_columns = [col for col in df.columns if 'time' in col.lower() or 'unix' in col.lower() or 'date' in col.lower()]
        
        if timestamp_columns:
            time_col = timestamp_columns[0]
            print(f"Found timestamp column: {time_col}")
            
            # Convert Unix timestamp to datetime
            df['datetime'] = pd.to_datetime(df[time_col], unit='s')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
        else:
            print("No timestamp column found. Please check the CSV structure.")
        
        # Drop any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Keep only OHLCV columns if they exist
        ohlcv_columns = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            matches = [c for c in df.columns if col in c.lower()]
            if matches:
                ohlcv_columns.append(matches[0])
        
        if ohlcv_columns:
            print(f"Using OHLCV columns: {ohlcv_columns}")
            df = df[ohlcv_columns]
            
            # Rename columns for consistency
            column_mapping = {}
            for col in ohlcv_columns:
                for std_name in ['open', 'high', 'low', 'close', 'volume']:
                    if std_name in col.lower():
                        column_mapping[col] = std_name.capitalize()
            
            df.rename(columns=column_mapping, inplace=True)
        
        # Resample to desired granularity if needed
        if granularity != '1min':
            df = DataHandler.resample_ohlcv(df, granularity)
        
        print("Data cleaned and resampled successfully")
        print(f"Final data shape: {df.shape}")
        return df
    
    @staticmethod
    def resample_ohlcv(df, granularity):
        """
        Resample OHLCV data to a new timeframe
        
        Parameters:
        - df: DataFrame with OHLCV data
        - granularity: target timeframe ('5min', '15min', etc.)
        """
        print(f"Resampling data to {granularity} timeframe...")
        
        # Convert granularity string to pandas offset string
        if granularity == '5min':
            offset = '5min'
        elif granularity == '15min':
            offset = '15min'
        else:
            offset = granularity
        
        # Resample OHLCV data
        resampled_df = df.resample(offset).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Drop any NaN rows after resampling
        resampled_df.dropna(inplace=True)
        
        return resampled_df


# ===============================================================================
# TECHNICAL INDICATORS
# ===============================================================================

class Indicators:
    """
    Collection of technical indicators used by strategies
    """
    @staticmethod
    def add_sma(df, period):
        """Add Simple Moving Average to a dataframe"""
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df, period):
        """Add Exponential Moving Average to a dataframe"""
        df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        """Add Relative Strength Index to a dataframe"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """Add MACD (Moving Average Convergence Divergence) to a dataframe"""
        df[f'EMA{fast_period}'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
        df[f'EMA{slow_period}'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = df[f'EMA{fast_period}'] - df[f'EMA{slow_period}']
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    
    @staticmethod
    def add_atr(df, period=14):
        """Add Average True Range to a dataframe"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(period).mean()
        return df
    
    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        """Add Bollinger Bands to a dataframe"""
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        df[f'BBands_Upper'] = df[f'SMA{period}'] + (df['Close'].rolling(window=period).std() * std_dev)
        df[f'BBands_Lower'] = df[f'SMA{period}'] - (df['Close'].rolling(window=period).std() * std_dev)
        return df


# ===============================================================================
# STRATEGY BASE CLASS
# ===============================================================================

class Strategy(abc.ABC):
    """
    Base Strategy class that all trading strategies will inherit from
    """
    def __init__(self, name, params=None):
        """
        Initialize strategy
        
        Parameters:
        - name: Name of the strategy
        - params: Dictionary of strategy parameters
        """
        self.name = name
        self.params = params or {}
    
    @abc.abstractmethod
    def add_indicators(self, df):
        """Add strategy-specific indicators to DataFrame"""
        pass
    
    @abc.abstractmethod
    def generate_signals(self, df):
        """Generate trading signals based on indicators"""
        pass
    
    def prepare_data(self, df):
        """Prepare data by adding indicators and generating signals"""
        df = self.add_indicators(df.copy())
        df = self.generate_signals(df)
        df.dropna(inplace=True)  # Remove rows with NaN values after adding indicators
        return df
    
    def get_signal_column(self):
        """Return the name of the signal column to be used in backtesting"""
        return f"{self.name}_Signal"


# ===============================================================================
# CONCRETE STRATEGY IMPLEMENTATIONS
# ===============================================================================

class SMAStrategy(Strategy):
    """
    Simple Moving Average crossover strategy
    """
    def __init__(self, params=None):
        """
        Initialize SMA crossover strategy
        
        Parameters:
        - params: Dictionary with 'fast_sma' and 'slow_sma' periods
        """
        super().__init__("SMA", params)
    
    def add_indicators(self, df):
        """Add SMA indicators to DataFrame"""
        fast_sma = self.params.get('fast_sma', 10)
        slow_sma = self.params.get('slow_sma', 50)
        
        df = Indicators.add_sma(df, fast_sma)
        df = Indicators.add_sma(df, slow_sma)
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals based on SMA crossovers
        
        Buy (1): Fast SMA crosses above Slow SMA
        Sell (-1): Fast SMA crosses below Slow SMA
        """
        fast_sma = self.params.get('fast_sma', 10)
        slow_sma = self.params.get('slow_sma', 50)
        signal_col = self.get_signal_column()
        
        # Create comparison column
        df[f'SMA{fast_sma}_gt_SMA{slow_sma}'] = df[f'SMA{fast_sma}'] > df[f'SMA{slow_sma}']
        
        # Create signals from changes in comparison
        df[signal_col] = df[f'SMA{fast_sma}_gt_SMA{slow_sma}'].diff()

        # Fill NaN with 0 to handle the first row or missing data
        df[signal_col] = df[signal_col].fillna(0)
        
        return df


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) strategy
    """
    def __init__(self, params=None):
        """
        Initialize RSI strategy
        
        Parameters:
        - params: Dictionary with 'rsi_period', 'rsi_oversold', and 'rsi_overbought'
        """
        super().__init__("RSI", params)
    
    def add_indicators(self, df):
        """Add RSI indicator to DataFrame"""
        rsi_period = self.params.get('rsi_period', 14)
        
        # Add SMAs for potential crossover filtering
        fast_sma = self.params.get('fast_sma', 10)
        slow_sma = self.params.get('slow_sma', 50)
        df = Indicators.add_sma(df, fast_sma)
        df = Indicators.add_sma(df, slow_sma)
        
        # Add RSI
        df = Indicators.add_rsi(df, rsi_period)
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals based on RSI values
        
        Buy (1): RSI drops below oversold level
        Sell (-1): RSI rises above overbought level
        """
        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        signal_col = self.get_signal_column()
        
        # Initialize signal column
        df[signal_col] = 0
        
        # Generate signals based on RSI thresholds
        df.loc[df['RSI'] < rsi_oversold, signal_col] = 1  # Oversold, potential buy
        df.loc[df['RSI'] > rsi_overbought, signal_col] = -1  # Overbought, potential sell
        
        return df


class CombinedStrategy(Strategy):
    """
    Combined strategy using both SMA crossover and RSI
    """
    def __init__(self, params=None):
        """
        Initialize combined SMA and RSI strategy
        
        Parameters:
        - params: Dictionary with SMA and RSI parameters
        """
        super().__init__("Combined", params)
        self.sma_strategy = SMAStrategy(params)
        self.rsi_strategy = RSIStrategy(params)
    
    def add_indicators(self, df):
        """Add both SMA and RSI indicators to DataFrame"""
        df = self.sma_strategy.add_indicators(df)
        df = self.rsi_strategy.add_indicators(df)
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals that require both SMA and RSI to confirm
        
        Buy (1): Fast SMA crosses above Slow SMA AND RSI below oversold level
        Sell (-1): Fast SMA crosses below Slow SMA AND RSI above overbought level
        """
        # Generate individual strategy signals
        df = self.sma_strategy.generate_signals(df)
        df = self.rsi_strategy.generate_signals(df)
        
        signal_col = self.get_signal_column()
        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        
        # Initialize combined signal column
        df[signal_col] = 0
        
        # Long when fast SMA crosses above slow SMA and RSI < oversold level
        df.loc[(df[self.sma_strategy.get_signal_column()] == 1) & 
               (df['RSI'] < rsi_oversold), signal_col] = 1
        
        # Short when fast SMA crosses below slow SMA and RSI > overbought level
        df.loc[(df[self.sma_strategy.get_signal_column()] == -1) & 
               (df['RSI'] > rsi_overbought), signal_col] = -1
        
        return df


# ===============================================================================
# STRATEGY FACTORY
# ===============================================================================

class StrategyFactory:
    """
    Factory class to create strategies based on their name
    """
    @staticmethod
    def create_strategy(strategy_name, params):
        """
        Create a strategy instance based on name
        
        Parameters:
        - strategy_name: Name of the strategy ('sma', 'rsi', 'combined', etc.)
        - params: Dictionary of strategy parameters
        """
        strategy_map = {
            'sma': SMAStrategy,
            'rsi': RSIStrategy,
            'combined': CombinedStrategy,
            # Add new strategies here
        }
        
        if strategy_name.lower() in strategy_map:
            return strategy_map[strategy_name.lower()](params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")


# ===============================================================================
# BACKTEST ENGINE
# ===============================================================================

class BacktestEngine:
    """
    Engine to run backtests and calculate performance metrics
    """
    def __init__(self, initial_capital=10000):
        """
        Initialize backtest engine
        
        Parameters:
        - initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
    
    def calculate_position_params(self, entry_price, current_capital, leverage, direction, 
                               risk_pct=2.0, position_size_pct=20.0, profit_pct=None):
        """
        Calculate position size, stop loss, and take profit levels based on risk management parameters
        
        Parameters:
        - entry_price: Entry price of the trade
        - current_capital: Current available capital
        - leverage: Leverage multiplier
        - direction: 1 for long, -1 for short
        - risk_pct: Maximum percentage of capital to risk per trade
        - position_size_pct: Percentage of capital to use for the position
        - profit_pct: Target profit percentage (if None, no take profit is set)
        
        Returns:
        - dict: Dictionary containing:
            - position_size: Dollar amount to commit to the position
            - stop_loss: Stop loss price level
            - take_profit: Take profit price level (or None)
            - risk_amount: Dollar amount at risk
            - potential_profit: Dollar amount of potential profit (or None)
        """
        # Calculate position size based on position_size_pct
        position_size = current_capital * (position_size_pct / 100)
        
        # Calculate max loss amount based on risk_pct
        max_loss_amount = current_capital * (risk_pct / 100)
        
        # Calculate price movement percentage that would cause max loss
        # Considering leverage and position size
        price_move_pct_for_max_loss = max_loss_amount / (position_size * leverage)
        
        # Calculate stop loss price based on direction
        if direction > 0:  # Long position
            stop_loss = entry_price * (1 - price_move_pct_for_max_loss)
        else:  # Short position
            stop_loss = entry_price * (1 + price_move_pct_for_max_loss)
        
        # Calculate take profit price if profit_pct is provided
        take_profit = None
        potential_profit = None
        if profit_pct is not None:
            # Calculate price movement percentage for target profit
            price_move_pct_for_target = profit_pct / (leverage * 100)
            
            if direction > 0:  # Long position
                take_profit = entry_price * (1 + price_move_pct_for_target)
            else:  # Short position
                take_profit = entry_price * (1 - price_move_pct_for_target)
            
            # Calculate potential profit amount
            potential_profit = position_size * leverage * (profit_pct / 100)
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': max_loss_amount,
            'potential_profit': potential_profit
        }
    
    def open_trade(self, results, i, position_direction, entry_price, current_capital, leverage, 
                  params, trade_id):
        """
        Open a new trading position and record it in the results dataframe
        
        Parameters:
        - results: DataFrame to record the trade in
        - i: Current row index in the dataframe
        - position_direction: Direction of the position (1 for long, -1 for short)
        - entry_price: Price at which to enter the position
        - current_capital: Available capital for the trade
        - leverage: Leverage multiplier
        - params: Dictionary with risk management parameters
        - trade_id: ID to assign to this trade
        
        Returns:
        - tuple: (updated trade_id, position_size, stop_loss, take_profit)
        """
        # Extract risk management parameters
        max_risk_pct = params.get('max_risk_pct', 2)
        position_size_pct = params.get('position_size_pct', 20)
        profit_target_pct = params.get('profit_target_pct', None)
        
        # Calculate position parameters
        pos_params = self.calculate_position_params(
            entry_price=entry_price,
            current_capital=current_capital,
            leverage=leverage,
            direction=position_direction,
            risk_pct=max_risk_pct,
            position_size_pct=position_size_pct,
            profit_pct=profit_target_pct
        )
        
        # Extract calculated values
        position_size = pos_params['position_size']
        stop_loss = pos_params['stop_loss']
        take_profit = pos_params['take_profit'] if pos_params['take_profit'] is not None else 0
        
        # Increment trade ID when opening a new trade
        trade_id += 1
        
        # Record trade information in results dataframe
        results.iloc[i, results.columns.get_loc('Position')] = position_direction
        results.iloc[i, results.columns.get_loc('Entry_Price')] = entry_price
        results.iloc[i, results.columns.get_loc('Trade_Id')] = trade_id
        results.iloc[i, results.columns.get_loc('Position_Size')] = position_size
        results.iloc[i, results.columns.get_loc('Risk_Amount')] = pos_params['risk_amount']
        results.iloc[i, results.columns.get_loc('Stop_Loss')] = stop_loss
        results.iloc[i, results.columns.get_loc('Take_Profit')] = take_profit
        
        return trade_id, position_size, stop_loss, take_profit
    
    def close_trade(self, results, i, position, position_size, entry_price, exit_price, leverage, trade_id, 
                   close_reason="signal"):
        """
        Close an existing trading position and record the P&L
        
        Parameters:
        - results: DataFrame to record the trade in
        - i: Current row index in the dataframe
        - position: Current position direction (1 for long, -1 for short)
        - position_size: Dollar size of the position
        - entry_price: Entry price of the position
        - exit_price: Exit price for the position
        - leverage: Leverage multiplier
        - trade_id: Current trade ID (will NOT be incremented)
        - close_reason: Reason for closing the trade ("signal", "stop_loss", "take_profit")
        
        Returns:
        - tuple: (trade_id, pnl, updated capital)
        """
        # Calculate P&L based on position direction and the ENTIRE trade (not just this candle)
        if position > 0:  # Long position
            entry_to_close_pct = (exit_price - entry_price) / entry_price
        else:  # Short position
            entry_to_close_pct = (entry_price - exit_price) / entry_price
        
        # Calculate P&L amount for the entire trade
        pnl = position_size * entry_to_close_pct * leverage
        
        # Record P&L in the results
        results.iloc[i, results.columns.get_loc('PnL')] = pnl
        
        # Determine trade result: 1 for win, -1 for loss, 0 for breakeven
        if pnl > 0:
            trade_result = 1  # Win
        elif pnl < 0:
            trade_result = -1  # Loss
        else:
            trade_result = 0  # Breakeven
            
        # Track trade result in the DataFrame
        results.iloc[i, results.columns.get_loc('Trade_Result')] = trade_result
        
        # Get current capital
        current_capital = results.iloc[i-1]['Capital'] + pnl
        
        # Record the same trade_id to mark the exit
        results.iloc[i, results.columns.get_loc('Trade_Id')] = trade_id
        
        # Reset position tracking variables in the results
        results.iloc[i, results.columns.get_loc('Position')] = 0
        results.iloc[i, results.columns.get_loc('Position_Size')] = 0
        results.iloc[i, results.columns.get_loc('Risk_Amount')] = 0
        results.iloc[i, results.columns.get_loc('Stop_Loss')] = 0
        results.iloc[i, results.columns.get_loc('Take_Profit')] = 0
        
        # Return updated values
        return trade_id, pnl, current_capital
    
    def run(self, df, strategy, params):
        """
        Run backtest for a given strategy and parameters
        
        Parameters:
        - df: DataFrame with price data
        - strategy: Strategy instance to test
        - params: Dictionary with strategy parameters
        
        Returns:
        - DataFrame with trading results
        - Dictionary with performance metrics
        """
        strategy_name = strategy.name
        leverage = params.get('leverage', 10)
        
        print(f"Backtesting {strategy_name} strategy with {leverage}x leverage...")
        
        # Create copy of data for results
        results = df.copy()
        signal_col = strategy.get_signal_column()
        
        # Initialize positions and account columns
        results['Position'] = 0
        results['Entry_Price'] = 0.0
        results['PnL'] = 0.0
        results['Trade_Id'] = 0
        results['Trade_Result'] = 0  # 1 for win, -1 for loss, 0 for breakeven
        results['Capital'] = float(self.initial_capital)  # Initialize as float explicitly
        # Convert Capital column to float dtype to prevent warnings
        results['Capital'] = results['Capital'].astype(float)
        results['Position_Size'] = 0.0  # Dollar value of the position
        results['Risk_Amount'] = 0.0  # Amount of capital at risk
        results['Stop_Loss'] = 0.0  # Stop loss price
        results['Take_Profit'] = 0.0  # Take profit price (if applicable)
        
        # Ensure we have a date column, or create one from the index
        if 'Date' not in results.columns:
            results = results.copy()  # Make a copy to avoid SettingWithCopyWarning
            results['Date'] = results.index.date
        
        # Track current position
        position = 0  # Direction (+1 for long, -1 for short)
        entry_price = 0
        trade_id = 0
        position_size = 0  # Dollar value of the position
        stop_loss = 0  # Stop loss price level
        take_profit = 0  # Take profit price level
        
        for i in range(1, len(results)):
            # Previous position carries forward by default
            results.iloc[i, results.columns.get_loc('Position')] = position
            
            # Get current signal
            signal = results.iloc[i-1][signal_col]  # Use previous bar's signal to enter this bar
            
            # Current capital (before this bar's trade)
            current_capital = results.iloc[i-1]['Capital']
            
            # Initialize PnL for this bar
            pnl = 0
            
            # Check for take profit hit (if active)
            take_profit_hit = False
            if position != 0 and take_profit != 0:
                if (position > 0 and results.iloc[i]['High'] >= take_profit) or \
                   (position < 0 and results.iloc[i]['Low'] <= take_profit):
                    # Close the trade with take profit price
                    exit_price = take_profit
                    trade_id, pnl, current_capital = self.close_trade(
                        results, i, position, position_size, entry_price, exit_price, 
                        leverage, trade_id, close_reason="take_profit"
                    )
                    take_profit_hit = True
                    position = 0
                    position_size = 0
                    stop_loss = 0
                    take_profit = 0
            
            # Check for stop loss hit (if position exists and take profit wasn't hit)
            stop_loss_hit = False
            if position != 0 and not take_profit_hit:
                if (position > 0 and results.iloc[i]['Low'] <= stop_loss) or \
                   (position < 0 and results.iloc[i]['High'] >= stop_loss):
                    # Close the trade with stop loss price
                    exit_price = stop_loss
                    trade_id, pnl, current_capital = self.close_trade(
                        results, i, position, position_size, entry_price, exit_price, 
                        leverage, trade_id, close_reason="stop_loss"
                    )
                    stop_loss_hit = True
                    position = 0
                    position_size = 0
                    stop_loss = 0
                    take_profit = 0
            
            # Handle position changes based on signal
            # Only take action if we have a valid signal (-1, 0, or 1) and no SL/TP was hit
            if signal != 0 and not (stop_loss_hit or take_profit_hit):
                # Case 1: We have a long position and get a sell signal
                if position > 0 and signal == -1:
                    # Close the long position
                    exit_price = results.iloc[i]['Close']
                    trade_id, pnl, current_capital = self.close_trade(
                        results, i, position, position_size, entry_price, exit_price, 
                        leverage, trade_id, close_reason="signal"
                    )
                    
                    # Open a short position with proper position sizing
                    trade_id, position_size, stop_loss, take_profit = self.open_trade(
                        results, i, -1, exit_price, current_capital, leverage, params, trade_id
                    )
                    position = -1
                    entry_price = exit_price
                    
                # Case 2: We have a short position and get a buy signal
                elif position < 0 and signal == 1:
                    # Close the short position
                    exit_price = results.iloc[i]['Close']
                    trade_id, pnl, current_capital = self.close_trade(
                        results, i, position, position_size, entry_price, exit_price, 
                        leverage, trade_id, close_reason="signal"
                    )
                    
                    # Open a long position with proper position sizing
                    trade_id, position_size, stop_loss, take_profit = self.open_trade(
                        results, i, 1, exit_price, current_capital, leverage, params, trade_id
                    )
                    position = 1
                    entry_price = exit_price
                    
                # Case 3: No position and get a buy signal
                elif position == 0 and signal == 1:
                    # Open a long position
                    entry_price = results.iloc[i]['Close']
                    trade_id, position_size, stop_loss, take_profit = self.open_trade(
                        results, i, 1, entry_price, current_capital, leverage, params, trade_id
                    )
                    position = 1
                    
                # Case 4: No position and get a sell signal
                elif position == 0 and signal == -1:
                    # Open a short position
                    entry_price = results.iloc[i]['Close']
                    trade_id, position_size, stop_loss, take_profit = self.open_trade(
                        results, i, -1, entry_price, current_capital, leverage, params, trade_id
                    )
                    position = -1
            
            # Calculate P&L for existing positions (when we're holding a position but don't have a signal to close)
            elif position != 0 and not (stop_loss_hit or take_profit_hit):
                if position > 0:  # Long position
                    # Calculate P&L as percentage change in price times position size times leverage
                    price_change_pct = (results.iloc[i]['Close'] - results.iloc[i-1]['Close']) / results.iloc[i-1]['Close']
                    pnl = position_size * price_change_pct * leverage
                else:  # Short position
                    # Calculate P&L as negative percentage change in price times position size times leverage
                    price_change_pct = (results.iloc[i-1]['Close'] - results.iloc[i]['Close']) / results.iloc[i-1]['Close']
                    pnl = position_size * price_change_pct * leverage
                
                results.iloc[i, results.columns.get_loc('PnL')] = pnl
            
            # Update capital
            results.iloc[i, results.columns.get_loc('Capital')] = float(current_capital) + float(results.iloc[i]['PnL'])
            
            # Check for 100% drawdown (or near total loss)
            if results.iloc[i]['Capital'] <= self.initial_capital * 0.01:
                print(f"Strategy failed with near total loss at bar {i}. Breaking test.")
                # Fill remaining rows with last capital value (effectively 0)
                results.loc[results.index[i:], 'Capital'] = results.iloc[i]['Capital']
                results.loc[results.index[i:], 'Position'] = 0  # Close position
                break
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(results, strategy_name, params)
        
        self._print_metrics(metrics)
        
        return results, metrics
    
    def _calculate_metrics(self, results, strategy_name, params):
        """
        Calculate performance metrics for a backtest result
        
        Parameters:
        - results: DataFrame with backtest results
        - strategy_name: Name of the strategy
        - params: Dictionary with strategy parameters
        
        Returns:
        - Dictionary with performance metrics
        """
        # Calculate performance metrics
        total_return_pct = ((results['Capital'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate daily returns
        results['Daily_Return'] = results['Capital'].pct_change()
        
        # Calculate Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = results['Daily_Return'].mean() / results['Daily_Return'].std() * (252 ** 0.5) if results['Daily_Return'].std() > 0 else 0  # Annualized
        
        # Calculate max drawdown
        results['Cummax'] = results['Capital'].cummax()
        results['Drawdown'] = (results['Capital'] - results['Cummax']) / results['Cummax'] * 100
        max_drawdown = results['Drawdown'].min()
        
        # Calculate trade statistics
        # Only count rows where Trade_Result is 0 or 1 (trade exits)
        completed_trades = results[results['Trade_Result'].isin([-1, 1])]
        total_completed_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['Trade_Result'] == 1])
        
        # Calculate win rate
        win_rate = winning_trades / total_completed_trades * 100 if total_completed_trades > 0 else 0
        
        # Calculate average trades per day
        start_date = results.index.min().date()
        end_date = results.index.max().date()
        days = (end_date - start_date).days
        days = max(1, days)  # Avoid division by zero
        trades_per_day = total_completed_trades / days if days > 0 else 0
        
        # Calculate trades per day for each day in the backtest
        results['Date'] = results.index.date
        daily_trades = completed_trades.groupby('Date').size()
        avg_trades_per_active_day = daily_trades.mean() if len(daily_trades) > 0 else 0
        
        # Strategy ID
        strategy_id = f"{strategy_name.lower()}_{params.get('granularity', '1min')}_fast{params.get('fast_sma', 10)}_slow{params.get('slow_sma', 50)}_rsi{params.get('rsi_period', 14)}_{params.get('rsi_oversold', 30)}_{params.get('rsi_overbought', 70)}_lev{params.get('leverage', 10)}"
        
        # Average profit per winning trade
        avg_win = completed_trades[completed_trades['Trade_Result'] == 1]['PnL'].mean() if winning_trades > 0 else 0
        
        # Average loss per losing trade
        losing_trades = completed_trades[completed_trades['Trade_Result'] == 0]
        avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate profit factor (sum of profits / sum of losses)
        total_profit = completed_trades[completed_trades['PnL'] > 0]['PnL'].sum()
        total_loss = abs(completed_trades[completed_trades['PnL'] < 0]['PnL'].sum())
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        # Create metrics dictionary
        metrics = {
            'Strategy': strategy_name.lower(),
            'Strategy_ID': strategy_id,
            'Leverage': params.get('leverage', 10),
            'Total_Return': total_return_pct,  # Percentage return
            'Final_Capital': results['Capital'].iloc[-1],  # Actual ending capital amount
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_completed_trades,
            'Winning_Trades': winning_trades, 
            'Losing_Trades': total_completed_trades - winning_trades,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Profit_Factor': profit_factor,
            'Trades_Per_Day': trades_per_day,
            'Trades_Per_Active_Day': avg_trades_per_active_day, 
            'Fast_SMA': params.get('fast_sma', 10),
            'Slow_SMA': params.get('slow_sma', 50),
            'RSI_Period': params.get('rsi_period', 14),
            'RSI_Oversold': params.get('rsi_oversold', 30),
            'RSI_Overbought': params.get('rsi_overbought', 70),
            'Granularity': params.get('granularity', '1min'),
        }
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Print summary metrics for a backtest"""
        print(f"Strategy: {metrics['Strategy']}")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${metrics['Final_Capital']:.2f}")
        print(f"Total Return: {metrics['Total_Return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2f}%")
        print(f"Win Rate: {metrics['Win_Rate']:.2f}%")
        print(f"Number of Trades: {metrics['Total_Trades']}")
        print(f"Trades Per Day: {metrics['Trades_Per_Day']:.2f}")
        print(f"Trades Per Active Day: {metrics['Trades_Per_Active_Day']:.2f}")


# ===============================================================================
# VISUALIZATION
# ===============================================================================

class Visualizer:
    """
    Class for visualizing backtest results
    """
    @staticmethod
    def plot_results(df, results, params, save_path=None):
        """
        Plot price chart with signals and performance
        
        Parameters:
        - df: DataFrame with price and indicator data
        - results: DataFrame with backtest results
        - params: Dictionary with strategy parameters
        - save_path: Path to save the plot (optional)
        """
        strategy = params.get('strategy', 'sma')
        fast_sma = params.get('fast_sma', 10)
        slow_sma = params.get('slow_sma', 50)
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        granularity = params.get('granularity', '1min')
        
        print(f"Plotting results for {strategy} strategy...")
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Plot price and moving averages
        ax1.plot(df.index, df['Close'], label='BTC Price', alpha=0.5, linewidth=1)
        ax1.plot(df.index, df[f'SMA{fast_sma}'], label=f'{fast_sma}-period SMA', linewidth=1)
        ax1.plot(df.index, df[f'SMA{slow_sma}'], label=f'{slow_sma}-period SMA', linewidth=1)
        
        # Plot buy and sell signals
        signal_col = f"{strategy}_Signal"
        
        buy_signals = df[df[signal_col] == 1]
        sell_signals = df[df[signal_col] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='^', alpha=1)
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Sell Signal', marker='v', alpha=1)
        
        # Format the price chart
        ax1.set_title(f'BTC Price with {strategy.upper()} Signals ({granularity})', fontsize=14)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot RSI
        ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
        ax2.axhline(y=rsi_oversold, color='green', linestyle='-', alpha=0.5)
        ax2.axhline(y=rsi_overbought, color='red', linestyle='-', alpha=0.5)
        ax2.fill_between(df.index, y1=rsi_oversold, y2=df['RSI'].where(df['RSI'] <= rsi_oversold), color='green', alpha=0.3)
        ax2.fill_between(df.index, y1=rsi_overbought, y2=df['RSI'].where(df['RSI'] >= rsi_overbought), color='red', alpha=0.3)
        ax2.set_title(f'RSI ({params.get("rsi_period", 14)})', fontsize=14)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot capital curve
        ax3.plot(results.index, results['Capital'], color='blue', linewidth=2)
        ax3.set_title('Account Capital', fontsize=14)
        ax3.set_ylabel('Capital (USD)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.close()


# ===============================================================================
# RESULTS MANAGEMENT
# ===============================================================================

class ResultsManager:
    """
    Class for managing and saving backtest results
    """
    @staticmethod
    def save_results_to_csv(all_metrics, results_data, csv_path='backtest_results.csv', detailed_path='detailed_results'):
        """
        Save backtest results to CSV files
        
        Parameters:
        - all_metrics: List of metric dictionaries from all backtests
        - results_data: Dictionary mapping strategy names to result dataframes
        - csv_path: Path to save summary metrics
        - detailed_path: Directory to save detailed results
        """
        print("Saving results to CSV...")
        
        # Create directory for detailed results if it doesn't exist
        os.makedirs(detailed_path, exist_ok=True)
        
        # Create summary DataFrame from metrics
        summary_df = pd.DataFrame(all_metrics)
        
        # Ensure columns are in a logical order
        column_order = [
            'Strategy', 'Strategy_ID', 'Leverage', 'Total_Return', 'Final_Capital', 
            'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate', 
            'Total_Trades', 'Winning_Trades', 'Trades_Per_Day', 'Trades_Per_Active_Day',
            'Fast_SMA', 'Slow_SMA', 'RSI_Period', 'RSI_Oversold', 'RSI_Overbought', 
            'Granularity'
        ]
        
        # Reorder columns if they exist in the dataframe
        available_columns = [col for col in column_order if col in summary_df.columns]
        summary_df = summary_df[available_columns]
        
        # Save summary to CSV with headers
        summary_df.to_csv(csv_path, index=True, header=True)
        print(f"Summary results saved to {csv_path}")
        
        # Save detailed results for each strategy with headers
        for strategy_name, result_df in results_data.items():
            file_path = os.path.join(detailed_path, f"{strategy_name}_detailed.csv")
            result_df.to_csv(file_path, index=True, header=True)
            print(f"Detailed results for {strategy_name} saved to {file_path}")

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def main():
    """
    Main function to run the analysis
    """
    print("Starting BTC trading strategy analysis...")
    
    # File path
    csv_file = 'btcusd_1-min_data.csv'
    
    # -----------------------------------------------------------------------
    # CONFIGURABLE PARAMETERS - Easily modify these to test different settings
    # -----------------------------------------------------------------------
    
    # Initial capital
    initial_capital = 200
    
    # Number of rows to load from the CSV (from the end)
    data_rows = 50000
    
    # -----------------------------------------------------------------------
    # STRATEGY-SPECIFIC PARAMETERS
    # -----------------------------------------------------------------------
    
    # Define parameters for each strategy separately for clarity
    sma_params = {
        'fast_sma': [ 9, 20],      # Fast SMA periods
        'slow_sma': [50, 100],    # Slow SMA periods
    }
    
    rsi_params = {
        'rsi_period': [7, 14, 21],             # RSI calculation period
        'rsi_oversold': [20, 30, 35],          # RSI oversold threshold (buy signal)
        'rsi_overbought': [65, 70, 80],        # RSI overbought threshold (sell signal)
    }
    
    # Combined strategy uses both SMA and RSI parameters
    
    # -----------------------------------------------------------------------
    # COMMON PARAMETERS (applied to all strategies)
    # -----------------------------------------------------------------------
    
    common_params = {
        # Timeframes to test
        # 'granularity': ['1min', '5min', '15min', '1h'],      # Data timeframes
        'granularity': ['5min'],
        
        # Risk and position sizing parameters
        'leverage': [10],                          # Trading leverage
        'max_risk_pct': [2.0],                     # Maximum risk per trade (% of capital)
        'position_size_pct': [80],      # Position size (% of capital)
        'profit_target_pct': [None, 6.0],        # Take profit target (% gain)
    }
    
    # -----------------------------------------------------------------------
    # FILTER SETTINGS - Control which strategies and parameters to test
    # -----------------------------------------------------------------------
    
    filters = {
        # Which strategies to test (comment out any you don't want to test)
        'strategies_to_test': [
            'sma',           # Simple Moving Average strategy
            'rsi',           # Relative Strength Index strategy
            'combined',      # Combined SMA + RSI strategy
        ],
        
        # Filter specific parameters (use None to test all values, or specify a single value)
        'granularity': None,    # Only test 5min timeframe (or None to test all)
        'leverage': None,           # Only test 10x leverage (or None to test all)
        'max_risk_pct': None,      # Only test 2% risk (or None to test all)
        'fast_sma': None,         # Test all fast SMA values
        'slow_sma': None,         # Test all slow SMA values
        'rsi_period': 14,         # Only test RSI period=14 (or None to test all)
        'rsi_oversold': 30,       # Only test RSI oversold=30 (or None to test all)
        'rsi_overbought': 70,     # Only test RSI overbought=70 (or None to test all)
        'position_size_pct': None, # Test all position sizing values
        'profit_target_pct': None, # Test all take profit values
    }
    
    # -----------------------------------------------------------------------
    # BUILD PARAMETER COMBINATIONS - Create test scenarios based on filters
    # -----------------------------------------------------------------------
    
    # Generate parameter combinations for each strategy type
    all_params = []
    
    # Helper function to apply filters to parameter values
    def apply_filter(param_name, param_values):
        if param_name in filters and filters[param_name] is not None:
            return [filters[param_name]]  # Use only the filtered value
        return param_values  # Use all values
    
    # Apply filters to common parameters
    filtered_common_params = {}
    for param_name, param_values in common_params.items():
        filtered_common_params[param_name] = apply_filter(param_name, param_values)
    
    # Generate SMA strategy parameters if enabled
    if 'sma' in filters['strategies_to_test']:
        # Apply filters to SMA parameters
        filtered_sma_params = {
            'strategy': ['sma'],
            'fast_sma': apply_filter('fast_sma', sma_params['fast_sma']),
            'slow_sma': apply_filter('slow_sma', sma_params['slow_sma']),
            # Add dummy RSI params for consistency (needed for ID generation but won't affect strategy)
            'rsi_period': [14],
            'rsi_oversold': [30],
            'rsi_overbought': [70],
            **filtered_common_params
        }
        
        # Generate all combinations for SMA
        sma_combinations = list(ParameterGrid(filtered_sma_params))
        all_params.extend(sma_combinations)
    
    # Generate RSI strategy parameters if enabled
    if 'rsi' in filters['strategies_to_test']:
        # Apply filters to RSI parameters
        filtered_rsi_params = {
            'strategy': ['rsi'],
            'rsi_period': apply_filter('rsi_period', rsi_params['rsi_period']),
            'rsi_oversold': apply_filter('rsi_oversold', rsi_params['rsi_oversold']),
            'rsi_overbought': apply_filter('rsi_overbought', rsi_params['rsi_overbought']),
            # Add required SMA params for trend identification (fast/slow SMA still useful in RSI strategy)
            'fast_sma': apply_filter('fast_sma', sma_params['fast_sma']),
            'slow_sma': apply_filter('slow_sma', sma_params['slow_sma']),
            **filtered_common_params
        }
        
        # Generate all combinations for RSI
        rsi_combinations = list(ParameterGrid(filtered_rsi_params))
        all_params.extend(rsi_combinations)
    
    # Generate Combined strategy parameters if enabled
    if 'combined' in filters['strategies_to_test']:
        # Combined strategy needs both SMA and RSI parameters
        filtered_combined_params = {
            'strategy': ['combined'],
            'fast_sma': apply_filter('fast_sma', sma_params['fast_sma']),
            'slow_sma': apply_filter('slow_sma', sma_params['slow_sma']),
            'rsi_period': apply_filter('rsi_period', rsi_params['rsi_period']),
            'rsi_oversold': apply_filter('rsi_oversold', rsi_params['rsi_oversold']),
            'rsi_overbought': apply_filter('rsi_overbought', rsi_params['rsi_overbought']),
            **filtered_common_params
        }
        
        # Generate all combinations for Combined
        combined_combinations = list(ParameterGrid(filtered_combined_params))
        all_params.extend(combined_combinations)
    
    print(f"Testing {len(all_params)} parameter combinations...")
    
    # Store all metrics and selected results
    all_metrics = []
    best_results = {}
    best_sharpe = -float('inf')
    best_return = -float('inf')
    
    # Create backtest engine
    backtest_engine = BacktestEngine(initial_capital=initial_capital)
    
    # Load data once (1min base data)
    print("Loading base 1min data once...")
    base_df = DataHandler.load_and_clean_data(csv_file, rows=data_rows, granularity='1min')
    
    # Group parameter combinations by granularity for efficient processing
    granularity_groups = {}
    for params in all_params:
        gran = params['granularity']
        if gran not in granularity_groups:
            granularity_groups[gran] = []
        granularity_groups[gran].append(params)
    
    # Process each granularity group
    for granularity, param_group in granularity_groups.items():
        print(f"\nProcessing {len(param_group)} combinations for {granularity} timeframe...")
        
        # Resample data for this granularity only once
        if granularity == '1min':
            df_for_granularity = base_df.copy()
        else:
            df_for_granularity = DataHandler.resample_ohlcv(base_df, granularity)
        
        # Skip if too few data points
        if len(df_for_granularity) < 100:
            print(f"Skipping {granularity} strategies due to insufficient data points")
            continue
            
        # Process each parameter combination with tqdm progress bar for this granularity
        
        # Create a custom file-like object to capture print statements
        class TqdmSafeFile:
            def __init__(self, original_stream):
                self.original_stream = original_stream
            
            def write(self, x):
                # Avoid print statements from corrupting the tqdm progress bar
                if len(x.rstrip()) > 0:
                    tqdm.write(x, end='', file=self.original_stream)
                    
            def flush(self):
                self.original_stream.flush()
        
        # Save the original stdout for restoring later
        original_stdout = sys.stdout
        
        # Redirect stdout to our custom handler to prevent tqdm bar issues
        sys.stdout = TqdmSafeFile(original_stdout)
        
        try:
            for params in tqdm(param_group, desc=f"Processing {granularity} combinations", unit="combination", 
                               position=0, leave=True, dynamic_ncols=True, smoothing=0.1):
                try:
                    # Create strategy 
                    strategy = StrategyFactory.create_strategy(params['strategy'], params)
                    
                    # Create strategy ID - include all relevant parameters
                    strategy_id = f"{params['strategy']}_{params['granularity']}"
                    strategy_id += f"_fast{params['fast_sma']}_slow{params['slow_sma']}"
                    
                    # Only include RSI parameters in the ID if it's an RSI or combined strategy
                    if params['strategy'] in ['rsi', 'combined']:
                        strategy_id += f"_rsi{params['rsi_period']}_{params['rsi_oversold']}_{params['rsi_overbought']}"
                    
                    strategy_id += f"_lev{params['leverage']}"
                    
                    # Add risk parameters to ID
                    if 'max_risk_pct' in params:
                        strategy_id += f"_risk{params['max_risk_pct']}"
                    
                    if 'position_size_pct' in params:
                        strategy_id += f"_pos{params['position_size_pct']}"
                    
                    if params.get('profit_target_pct') is not None:
                        strategy_id += f"_tp{params['profit_target_pct']}"
                    
                    # Use the already resampled data for this granularity
                    df = df_for_granularity.copy()
                    
                    # Prepare data with indicators and signals
                    df = strategy.prepare_data(df)
                    
                    # Backtest strategy
                    results, metrics = backtest_engine.run(df, strategy, params)
                    
                    # Add strategy ID to metrics
                    metrics['Strategy_ID'] = strategy_id
                    all_metrics.append(metrics)
                    
                    # Save best strategies by different metrics
                    if metrics['Sharpe_Ratio'] > best_sharpe:
                        best_sharpe = metrics['Sharpe_Ratio']
                        best_results[f'best_sharpe_{strategy_id}'] = results
                    
                    if metrics['Total_Return'] > best_return:
                        best_return = metrics['Total_Return']
                        best_results[f'best_return_{strategy_id}'] = results
                    
                    # Generate plot for the top strategies only
                    if metrics['Sharpe_Ratio'] > 1.0 and metrics['Total_Return'] > 50:
                        plot_path = f"plots/BTC_{strategy_id}_backtest.png"
                        os.makedirs('plots', exist_ok=True)
                        Visualizer.plot_results(df, results, params, save_path=plot_path)
                        
                except Exception as e:
                    tqdm.write(f"Error with parameter combination {params}: {e}")
        finally:
            # Restore original stdout when done
            sys.stdout = original_stdout
    
    # Save all results to CSV
    ResultsManager.save_results_to_csv(all_metrics, best_results)
    
    # Display top 5 strategies by Sharpe ratio
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Sort by metrics and display top results
        print("\n===== BEST STRATEGIES BY DIFFERENT METRICS =====")
        
        # Top 5 by Sharpe Ratio (risk-adjusted returns)
        top_by_sharpe = metrics_df.sort_values('Sharpe_Ratio', ascending=False).head(5)
        print("\nTop 5 Strategies by Sharpe Ratio:")
        print(top_by_sharpe[['Strategy_ID', 'Sharpe_Ratio', 'Total_Return', 'Final_Capital', 'Win_Rate', 'Max_Drawdown', 'Total_Trades']])
        
        # Top 5 by Total Return (absolute returns)
        top_by_return = metrics_df.sort_values('Total_Return', ascending=False).head(5)
        print("\nTop 5 Strategies by Total Return:")
        print(top_by_return[['Strategy_ID', 'Total_Return', 'Final_Capital', 'Sharpe_Ratio', 'Win_Rate', 'Max_Drawdown', 'Total_Trades']])
        
        # Top 5 by Win Rate (consistency)
        top_by_winrate = metrics_df.sort_values('Win_Rate', ascending=False).head(5)
        print("\nTop 5 Strategies by Win Rate:")
        print(top_by_winrate[['Strategy_ID', 'Win_Rate', 'Total_Return', 'Sharpe_Ratio', 'Final_Capital', 'Max_Drawdown', 'Total_Trades']])
        
        # Top 5 by Profit Factor (if available)
        if 'Profit_Factor' in metrics_df.columns:
            top_by_profit_factor = metrics_df.sort_values('Profit_Factor', ascending=False).head(5)
            print("\nTop 5 Strategies by Profit Factor:")
            print(top_by_profit_factor[['Strategy_ID', 'Profit_Factor', 'Total_Return', 'Sharpe_Ratio', 'Win_Rate', 'Max_Drawdown', 'Total_Trades']])
        
        # Best strategy with lowest drawdown (among profitable strategies)
        profitable = metrics_df[metrics_df['Total_Return'] > 0]
        if not profitable.empty:
            best_drawdown = profitable.sort_values('Max_Drawdown', ascending=False).head(5)
            print("\nBest Strategies with Lowest Drawdown (among profitable):")
            print(best_drawdown[['Strategy_ID', 'Max_Drawdown', 'Total_Return', 'Sharpe_Ratio', 'Win_Rate', 'Total_Trades']])
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()