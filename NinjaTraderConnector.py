import platform
import ctypes
from ctypes import wintypes
import time

# Determine Python architecture
architecture = platform.architecture()[0]

# Set DLL path based on architecture
if architecture == '64bit':
    dll_path = 'C:\\Program Files\\NinjaTrader 8\\bin64\\NtDirect.dll'
else:
    dll_path = 'C:\\Program Files\\NinjaTrader 8\\bin\\NtDirect.dll'

# Load the ntDirect.dll
try:
    ntdirect = ctypes.WinDLL(dll_path)
except OSError as e:
    print(f"Error loading {dll_path}: {e}")
    raise

# Define return types for the functions we'll use
ntdirect.OrderStatus.restype = ctypes.c_double
ntdirect.MarketData.restype = ctypes.c_double

# Function to get market data
def get_market_data(instrument, data_type):
    market_data_function = ntdirect.MarketData
    market_data_function.argtypes = [wintypes.LPCSTR, wintypes.LPCSTR]
    result = market_data_function(ctypes.c_char_p(instrument.encode('utf-8')),
                                  ctypes.c_char_p(data_type.encode('utf-8')))
    return result

# Function to display market data for NQ 09-24
def display_market_data():
    instrument = 'NQ 09-24'  # The instrument symbol for SEP 24 NQ
    bid = get_market_data(instrument, 'bid')
    ask = get_market_data(instrument, 'ask')
    last = get_market_data(instrument, 'last')
    print(f"Instrument: {instrument}")
    print(f"Bid: {bid}")
    print(f"Ask: {ask}")
    print(f"Last: {last}")

if __name__ == "__main__":
    while True:
        display_market_data()
        time.sleep(1)  # Delay for 1 second
