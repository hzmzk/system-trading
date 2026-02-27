import pandas as pd
import os
import yfinance as yf
import csv

from multiprocessing import Pool
from datetime import datetime, timedelta

from util import datetime_csv, partition_list,  price_normalization

def update_database():
    create_sector_industries_data()
    create_top_companies_data()
    update_risk_free_rate()
    remove_empty()

    every_price = os.listdir("price_data/")
    every_sector = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"]
    
    Pool().map(update_price_data, every_price)
    Pool().map(create_industry_normalization_data, every_sector)

#return list of industries in a sector
def sector_industries_list(sector):
    with open("sector_industries/" + sector + ".csv") as myfile:
        reader = csv.reader(myfile)
        mylist = list(reader)
    return mylist[0]

#return list of top companies in an industry
def industry_top_companies_list(industry):
    with open("top_companies/" + industry + ".csv") as myfile:
        reader = csv.reader(myfile)
        mylist = list(reader)
    return mylist[0]

def create_sector_industries_data():
     sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"]    
     for sector in sector_list:
          industry_list = yf.Sector(sector).industries.index
          with open("sector_industries/" + sector + ".csv", 'w') as myfile:
               wr = csv.writer(myfile)
               wr.writerow(industry_list)

def create_top_companies_data():
     sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities", "healthcare", "industrials", "real-estate", "technology"]    
     for sector in sector_list:
          industry_list = sector_industries_list(sector)
          for industry in industry_list:
               top_companies = yf.Industry(industry).top_companies
               if(top_companies is not None):
                    with open("top_companies/" + industry + ".csv", 'w') as myfile:
                         wr = csv.writer(myfile)
                         wr.writerow(top_companies.index.to_list())

def create_price_data(stock_list):
    for stock in stock_list:    
        price = yf.download([stock], period="max", auto_adjust = True, progress=False)["Close"]
        price.to_csv("price_data/" + stock + ".csv")

def create_industry_data(sector_list):
    for sector in sector_list:
        industry_list = sector_industries_list(sector)
        for industry in industry_list:
            if(industry+".csv" in os.listdir("top_companies/")):
                stock_list = industry_top_companies_list(industry)
                for stock in stock_list:    
                    price = yf.download([stock], period="max", auto_adjust = True, progress=False)["Close"]
                    price.to_csv("price_data/" + stock + ".csv")
    print(sector_list, " done")

def create_industry_normalization_data(sector, start_date="2010"):

    industry_list = sector_industries_list(sector)
    multi_agg_norm_price = pd.DataFrame()

    for industry in industry_list:
        if(industry+".csv" in os.listdir("top_companies/")):
            stock_list = industry_top_companies_list(industry)

            stock_list_csv = [stock + ".csv" for stock in stock_list]
            if(not set(stock_list_csv).issubset(set(os.listdir("price_data/")))):
                create_price_data(stock_list)
            
            multi_norm_price = pd.DataFrame()
            for stock in stock_list:  
                price = datetime_csv("price_data/" + stock + ".csv", start=start_date)
                normalized_price = price_normalization(price)
                multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

            agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    

            agg_norm_price = pd.DataFrame(agg_norm_price, columns=[industry])
            agg_norm_price.to_csv("industry_normalization_price/" + industry + ".csv")

def update_price_data(stock_csv):
    price = datetime_csv("price_data/" + stock_csv)
    last_date = price.index[-1:].item()
    stock = price.columns.item()
    new_price = yf.download([stock], start=last_date, auto_adjust = True, progress=False)["Close"]
    price = price.iloc[:-1]
    price = pd.concat([price, new_price])
    price.to_csv("price_data/" + stock + ".csv")

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
            pd.read_csv(stock)        
        except:
            os.remove("price_data/" + stock)


