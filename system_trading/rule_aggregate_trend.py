import pandas as pd
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from csv_dataset import sector_industries_list, industry_top_companies_list
from rule_ewmac import multi_ewmac
from util import datetime_csv, price_normalization
from backtest import *
from trade_class import Stock
 

def industry_trend_rule(stock, start_date="2010", backtest_mode=False):
    ticker = yf.Ticker(stock)
    industry = ticker.info.get('industryKey')

    industry_norm_price = datetime_csv("industry_normalization_price/" + industry + ".csv", start=start_date)

    if(not backtest_mode):
        forecast = multi_ewmac(industry_norm_price, parameter="normal")
    else:
        forecast = multi_ewmac(industry_norm_price, parameter="normal", backtest_mode=True)
    
    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
        industry_name = industry_norm_price.columns.item()
        forecast[forecast[industry_name] > 20] = 20
        forecast[forecast[industry_name] < -20] = -20
        forecast = forecast.rename(columns={industry_name:stock})

    return forecast


def industry_trend_list(momentum_value, start_date="2024", show_graph = False):
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    passed_sector_list = []
    passed_industry_list = []
    for sector in sector_list:
        sector_print_trigger = True
        industry_list =  sector_industries_list(sector)
        for industry in industry_list:
            if(industry + ".csv" not in os.listdir("industry_normalization_price/")):
                continue
            industry_norm_price = datetime_csv("industry_normalization_price/" + industry + ".csv", start=start_date)
            industry_aggregate_momentum = multi_ewmac(industry_norm_price, parameter="normal")
            if(industry_aggregate_momentum > momentum_value):
                passed_industry_list.append(industry)
                if(sector_print_trigger):
                    passed_sector_list.append(sector)
                    print(sector)
                    sector_print_trigger = False
                print(industry, ":", industry_aggregate_momentum)
                if(show_graph):
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.plot(industry_norm_price)
                    plt.tight_layout()
                    plt.show()
        if(sector_print_trigger == False):
            print('\n')
    print(passed_sector_list)
    print(passed_industry_list)


# Check trend forecast directly from yahoo finance
def industry_trend_check(industry_key, start_date="2024-01-01"):
    stock_list = yf.Industry(industry_key).top_companies.index
    multi_norm_price = pd.DataFrame()

    for stock in stock_list:    
        price = yf.download([stock], start=start_date, auto_adjust = True, progress=False)["Close"]
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = pd.DataFrame({industry_key:(multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()}) 
    agg_momentum_forecast = multi_ewmac(agg_norm_price, parameter="normal")

    return agg_momentum_forecast



# Plot aggregate normalization using data extracted directly from yahoo finance
def plot_agg_norm(industry_key, start_date="2024-01-01", display_all=True):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title(industry_key)
    multi_norm_price = pd.DataFrame()
    stock_list = yf.Industry(industry_key).top_companies.index
    for stock in stock_list:    
        price = yf.download([stock], start=start_date, auto_adjust = True, progress=False)["Close"]
        if(display_all):     
            ax.plot(price_normalization(price), '--')
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()
    ax.plot(agg_norm_price, color="black")
    
    plt.tight_layout()
    plt.show()

#######################################################################################################

def dr_industry_trend_rule(stock):
    price = Stock(stock).price
    
    forecast = industry_trend_rule(stock, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return).mean()
    return sample_return

def sr_industry_trend_rule(stock):
    price = Stock(stock).price

    forecast = industry_trend_rule(stock, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    index = common_index([price, forecast])
    price = price.loc[index]
    forecast = forecast.loc[index]

    return sharpe_ratio(price, forecast)

def bt_industry_trend_rule(price):
    price = Stock(stock).price

    forecast = industry_trend_rule(stock, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    
    return p_value_bootstrap(price, forecast)

def mc_industry_trend_rule(price):
    price = Stock(stock).price

    forecast = industry_trend_rule(stock, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    
    return p_value_montecarlo(price, forecast)
