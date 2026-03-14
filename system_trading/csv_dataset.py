import pandas as pd
import os
import yfinance as yf
import csv
import json

from multiprocessing import Pool
from datetime import datetime, timedelta

from trade_class import Stock
from util import datetime_csv, multi_datetime_csv, partition_list,  price_normalization


def update_list():
    create_industry_in_sector()
    create_industry_basket()
    
def update_database():
    remove_empty()
    update_risk_free_rate()

    every_price = [ i[:-4] for i in os.listdir("price_data/") ]
    Pool().map(update_price_data, every_price)

    with open('other_data/industry_basket.json') as f:
        every_basket = json.load(f).keys()
    Pool().map(create_aggregate_normalization_price, every_basket)


###########################################################################################################################################

def create_industry_in_sector():
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"]    
    sector_industry_dict = {}
    for sector in sector_list:
        industry_list = yf.Sector(sector).industries.index.to_list()
        sector_industry_dict[sector] = industry_list 
    with open("other_data/industry_in_sector.json", 'w') as f:
        json.dump(sector_industry_dict,f)

def create_industry_basket():
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"]    
    industry_company_dict = {}
    for sector in sector_list:
        industry_list = sector_industries_list(sector)
        for industry in industry_list:
            top_companies = yf.Industry(industry).top_companies
            if(top_companies is not None):
                industry_company_dict[industry] = top_companies.index.to_list()

    with open("other_data/industry_basket.json", 'w') as f:
        json.dump(industry_company_dict,f)

#return list of industries in a sector
def sector_industries_list(sector):
    with open("other_data/industry_in_sector.json") as f:
        industry_list = json.load(f)[sector]
    return industry_list

#return list of top companies in an industry
def industry_basket(industry):
    with open("other_data/industry_basket.json") as f:
        company_list = json.load(f)[industry]
    return company_list

###########################################################################################################################################

def create_price_data(ticker_list):
    for ticker in ticker_list:    
        price = yf.download([ticker], period="max", auto_adjust = True, progress=False)
        price.to_csv("price_data/" + ticker + ".csv")

def update_price_data(ticker):
    price = multi_datetime_csv("price_data/" + ticker + ".csv")
    last_date = price.index[-1:].item()
    new_price = yf.download([ticker], start=last_date, auto_adjust = True, progress=False)
    price = price.iloc[:-1]
    price = pd.concat([price, new_price])
    price.to_csv("price_data/" + ticker + ".csv")

############################################################################################################################################

def create_aggregate_normalization_price(basket_name, year_start="2020"):
    with open('other_data/industry_basket.json') as f:
        stock_list = json.load(f)[basket_name]
    
    multi_agg_norm_price = pd.DataFrame()

    stock_list_csv = [stock + ".csv" for stock in stock_list]
    if(not set(stock_list_csv).issubset(set(os.listdir("price_data/")))):
        create_price_data(stock_list)
    
    multi_norm_price = pd.DataFrame()
    for stock in stock_list:
        price = Stock(stock, start_date=year_start).price
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    

    agg_norm_price = pd.DataFrame(agg_norm_price, columns=[basket_name])
    agg_norm_price.to_csv("aggregate_normalization_price/" + basket_name + ".csv")


###############################################################################################################################################

def update_risk_free_rate():
    rf = datetime_csv("other_data/^IRX.csv")
    last_date = rf.index[-1:].item()
    new_rf = yf.download(["^IRX"], start=last_date, auto_adjust = True, progress=False)["Close"]
    rf = rf.iloc[:-1]
    rf = pd.concat([rf, new_rf])
    rf.to_csv("other_data/^IRX.csv")

def remove_empty():
    stock_list = [i for i in os.listdir("price_data/")]
    for stock in stock_list:
        try:
            pd.read_csv("price_data/" + stock)    
        except:
            os.remove("price_data/" + stock)

