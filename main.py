# Import necessary libraries
import math
import ccxt
import pandas as pd
import mplfinance as mpf
import time

# Initialize the BingX exchange
exchange = ccxt.bingx()
# exchange = ccxt.bitstamp()

# Load markets to ensure that the exchange's markets are initialized
exchange.load_markets()

# Function to fetch and prepare data
def get_ohlcv_data(symbol, timeframe, limit):
    # Ensure the exchange has the symbol and timeframe
    if symbol not in exchange.symbols:
        raise ValueError(f"Symbol {symbol} not available on BingX.")
    if timeframe not in exchange.timeframes:
        raise ValueError(f"Timeframe {timeframe} not available on BingX.")
    
    # Fetch OHLCV data
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=None, limit=limit)
    except Exception as e:
        print(f"An error occurred while fetching {symbol} data: {e}")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Convert UTC to IST
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# Define symbols (commodities and currency pairs)
# symbols = ['BTC/USDT', 'EUR/USDT', 'JPY/USDT', 'GBP/USDT', 'XAU/USDT', 'USOIL/USDT']

# Define time frames (you can modify this list)
# timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']

symbols = ['BTC/USDT']
timeframes = ['1m', '5m', '15m', '1h']

# Number of data points to fetch
limit = 150

# Function to plot multiple symbols and timeframes
def plot_symbols_timeframes(symbols, timeframes):
    total_plots = len(symbols) * len(timeframes)
    cols = 2  # Number of columns in the subplot grid
    rows = math.ceil(total_plots / cols)

    # Adjust figure size dynamically
    figure_width = 20  # Fixed width in inches
    figure_height_per_row = 5  # Height per row in inches
    figure_height = rows * figure_height_per_row
    figure_size = (figure_width, figure_height)

    # Create a custom style with adjusted font sizes
    custom_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        rc={
            'font.size': 10,            # General font size
            'axes.labelsize': 12,       # Axis label font size
            'xtick.labelsize': 8,       # X-axis tick label font size
            'ytick.labelsize': 8,       # Y-axis tick label font size
            'figure.titlesize': 14      # Figure title font size
        }
    )

    # Create the figure
    fig = mpf.figure(figsize=figure_size, style=custom_style)
    axs = []

    # Create subplots dynamically
    for i in range(1, total_plots + 1):
        ax = fig.add_subplot(rows, cols, i)
        axs.append(ax)

    idx = 0  # Index for accessing axs
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Fetching data for {symbol} at interval {timeframe}...")
            df = get_ohlcv_data(symbol, timeframe, limit)
            
            if df.empty:
                print(f"No data fetched for {symbol} at interval {timeframe}.\n")
                idx += 1
                continue
            
            print(f"Plotting chart for {symbol} at interval {timeframe}...")
            
            # Plot on the specified axes
            mpf.plot(
                df,
                type='candle',
                ax=axs[idx],
                style=custom_style,
                volume=False,  # Set to True if you want to include volume and pass volume axes
                axtitle=f'{symbol} {timeframe}',
                ylabel='Price',
                show_nontrading=False
            )
            print(f"Chart plotted for {symbol} at interval {timeframe}.\n")
            idx += 1
            # Pause to avoid rate limits
            time.sleep(exchange.rateLimit / 1000)

    # Hide any unused subplots
    total_subplots = rows * cols
    if idx < total_subplots:
        for i in range(idx, total_subplots):
            fig.delaxes(axs[i])

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Display and save the figure
    # mpf.show()  # Display the figure
    fig.savefig('Combined_Symbols_Timeframes.png')

    print("Combined chart saved as Combined_Symbols_Timeframes.png")

# Call the function to plot symbols and timeframes
plot_symbols_timeframes(symbols, timeframes)