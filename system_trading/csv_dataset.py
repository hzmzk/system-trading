import pandas as pd
import os
import yfinance as yf

from multiprocessing import Pool
from datetime import datetime, timedelta

from util import datetime_csv, partition_list
from trade_function import price_normalization


def update_database(parameter="all"):
    every_price = partition_list(os.listdir("price_data/"))
    every_sector = partition_list(["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"])
    
    if(parameter=="price"):
        Pool().map(update_price_data, every_price)
    elif(parameter=="industry_norm"):
        Pool().map(create_industry_normalization_data, every_sector)
    elif(parameter=="all"):
        Pool().map(update_price_data, every_price)
        Pool().map(create_industry_normalization_data, every_sector)

def update_price_data(stock_list):
    for stock_csv in stock_list:
        price = datetime_csv("price_data/" + stock_csv)
        last_date = price.index[-1:].item()
        stock = price.columns.item()
        new_price = yf.download([stock], start=last_date, auto_adjust = True, progress=False)["Close"]
        price = price.iloc[:-1]
        price = pd.concat([price, new_price])
        price.to_csv("price_data/" + stock + ".csv")

def create_price_data(stock_list):
    for stock in stock_list:    
        price = yf.download([stock], period="max", auto_adjust = True, progress=False)["Close"]
        price.to_csv("price_data/" + stock + ".csv")

def create_industry_data(sector_list):
    for sector in sector_list:
        industry_list = yf.Sector(sector).industries.index

        for industry in industry_list:
            if(yf.Industry(industry).top_companies is not None):
                stock_list = yf.Industry(industry).top_companies.index

                for stock in stock_list:    
                    price = yf.download([stock], period="max", auto_adjust = True, progress=False)["Close"]
                    price.to_csv("price_data/" + stock + ".csv")
    print(sector_list, " done")

def create_industry_normalization_data(sector_list):
    for sector in sector_list:
        industry_list = yf.Sector(sector).industries.index
        multi_agg_norm_price = pd.DataFrame()

        for industry in industry_list:
            if(yf.Industry(industry).top_companies is not None):
                stock_list = yf.Industry(industry).top_companies.index

                stock_list_csv = [stock + ".csv" for stock in stock_list]
                if(not set(stock_list_csv).issubset(set(os.listdir("price_data/")))):
                    create_price_data(stock_list)
                
                multi_norm_price = pd.DataFrame()
                for stock in stock_list:  
                    price = datetime_csv("price_data/" + stock + ".csv", start="2024-01-01")
                    normalized_price = price_normalization(price)
                    multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

                agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    

                agg_norm_price = pd.DataFrame(agg_norm_price, columns=[industry])
                multi_agg_norm_price = multi_agg_norm_price.join(agg_norm_price, how="outer")

        multi_agg_norm_price.to_csv("industry_normalization_price/" + sector + ".csv")
    print(sector_list, "done")


def to_date(x):
    date = str(x)
    date = '-'.join([date[:4], date[4:6], date[6:]])
    return date

def fama_risk_free_rate(fama_daily_rf_csv):
    daily = pd.read_csv(fama_daily_rf_csv)

    date = [to_date(i) for i in daily["Unnamed: 0"]]
    date_index = pd.DatetimeIndex(date)

    risk_free = pd.DataFrame({"rf":daily["RF"].to_list()}, index=date_index)
    risk_free.to_csv("other_data/" + "FAMA_rf" + ".csv")


