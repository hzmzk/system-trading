import pandas as pd
import yfinance as yf

def get_quote(stock_name):
    return yf.download([stock_name], period = '5d', auto_adjust = True, progress=False)["Close"]