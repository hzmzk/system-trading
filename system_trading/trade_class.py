import yfinance as yf
import pandas as pd
import os

from util import datetime_csv

class Stock:
    def __init__(self, ticker_symbol, start_date="", end_date=""):
 
        if(not ticker_symbol + ".csv" in os.listdir("price_data/")):
            price =  yf.download([ticker_symbol], period="max", auto_adjust=True, progress=False)["Close"]
            price.to_csv("price_data/" + stock + ".csv")

        self.ticker = ticker_symbol
        self.price = datetime_csv("price_data/" + ticker_symbol + ".csv", start=start_date, end=end_date)  