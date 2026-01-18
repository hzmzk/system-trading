import yfinance as yf
import pandas as pd

class Stock:
    def __init__(self, ticker, horizon='1y'):
        self.price = yf.download([ticker], period = horizon, auto_adjust = True, progress=False)["Close"]
    