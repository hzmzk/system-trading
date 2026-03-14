import yfinance as yf
import pandas as pd
import os
import json

from util import multi_datetime_csv

class Stock:
    def __init__(self, ticker_symbol, start_date="", end_date=""):

        with open('other_data/stock_data.json') as f:
            stock_data = json.load(f)
        
        if(not ticker_symbol in stock_data.keys()):
            stock_data[ticker_symbol] = {}
            with open('other_data/stock_data.json', 'w') as f:
                json.dump(stock_data, f)
 
        if(not ticker_symbol + ".csv" in os.listdir("price_data/")):
            price = yf.download([ticker_symbol], period="max", auto_adjust = True, progress=False)
            price.to_csv("price_data/" + ticker_symbol + ".csv")

        if(not 'industry' in stock_data[ticker_symbol].keys()):
            industry = yf.Ticker(ticker_symbol).info.get("industryKey")
            stock_data[ticker_symbol]["industry"] = industry
            
            with open('other_data/stock_data.json', 'w') as f:
                json.dump(stock_data, f)

        if(not 'sector' in stock_data[ticker_symbol].keys()):
            sector = yf.Ticker(ticker_symbol).info.get("sectorKey")
            stock_data[ticker_symbol]["sector"] = sector

            with open('other_data/stock_data.json', 'w') as f:
                json.dump(stock_data, f)

        self.ticker = ticker_symbol
        self.price = multi_datetime_csv("price_data/" + ticker_symbol + ".csv", parameter=["Close"], start=start_date, end=end_date)["Close"]  
        self.open = multi_datetime_csv("price_data/" + ticker_symbol + ".csv", parameter=["Open"], start=start_date, end=end_date)["Open"]  
        self.industry = stock_data[ticker_symbol]["industry"]
        self.sector = stock_data[ticker_symbol]["sector"]