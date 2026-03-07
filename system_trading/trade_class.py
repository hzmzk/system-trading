import yfinance as yf
import pandas as pd
import os
import pickle

from util import datetime_csv

class Stock:
    def __init__(self, ticker_symbol, start_date="", end_date=""):

        f_pickle = open("other_data/stock_data.pkl", "rb")
        stock_data = pickle.load(f_pickle)
 
        if(not ticker_symbol + ".csv" in os.listdir("price_data/")):
            price =  yf.download([ticker_symbol], period="max", auto_adjust=True, progress=False)["Close"]
            price.to_csv("price_data/" + ticker_symbol + ".csv")

        if(not ticker_symbol in stock_data["industry"].keys()):
            industry = yf.Ticker(ticker_symbol).info.get("industryKey")
            stock_data["industry"][ticker_symbol] = industry
            f_pickle = open("other_data/stock_data.pkl", "wb")
            pickle.dump(stock_data, f_pickle)

        if(not ticker_symbol in stock_data["sector"].keys()):
            sector = yf.Ticker(ticker_symbol).info.get("sectorKey")
            stock_data["sector"][ticker_symbol] = sector
            f_pickle = open("other_data/stock_data.pkl", "wb")
            pickle.dump(stock_data, f_pickle)

        self.ticker = ticker_symbol
        self.price = datetime_csv("price_data/" + ticker_symbol + ".csv", start=start_date, end=end_date)  
        self.industry = stock_data["industry"][ticker_symbol]
        self.sector = stock_data["sector"][ticker_symbol]