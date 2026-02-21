import pandas as pd
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt

from rule_ewmac import multi_ewmac
from util import datetime_csv


def industry_trend_rule(stock, backtest_mode=False):
    ticker = yf.Ticker(stock)
    sector = ticker.info.get('sectorKey')
    industry = ticker.info.get('industryKey')

    multi_industry_norm_price = datetime_csv("industry_normalization_price/" + sector + ".csv", start="2024-01-01")
    industry_norm_price = pd.DataFrame({industry:multi_industry_norm_price[industry]})

    if(not backtest_mode):
        forecast = multi_ewmac(industry_norm_price, parameter="normal")
    else:
        forecast = multi_ewmac(industry_norm_price, parameter="normal", backtest_mode=True)
    
    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
        column_name = industry_norm_price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20

    return forecast


def industry_trend_list(momentum_value, show_graph = False):
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    passed_sector_list = []
    passed_industry_list = []
    for sector in sector_list:
        sector_print_trigger = True
        multi_industry_norm_price = datetime_csv("industry_normalization_price/" + sector + ".csv", start="2024-01-01")
        for industry in multi_industry_norm_price.columns:
            industry_norm_price = pd.DataFrame({industry:multi_industry_norm_price[industry]})
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
                    ax.plot(multi_industry_norm_price[industry])
                    plt.tight_layout()
                    plt.show()
        if(sector_print_trigger == False):
            print('\n')
    print(passed_sector_list)
    print(passed_industry_list)


# Check trend forecast directly from yahoo finance
def industry_trend_check(industry_key, horizon="2y"):
    stock_list = yf.Industry(industry_key).top_companies.index
    multi_norm_price = pd.DataFrame()

    for stock in stock_list:    
        price = yf.download([stock], period=horizon, auto_adjust = True, progress=False)["Close"]
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    
    agg_momentum_forecast = multi_ewmac(agg_norm_price, parameter="normal")

    return agg_momentum_forecast


# Plot aggregate normalization using data extracted directly from yahoo finance
def plot_agg_norm(stock_list, display_all=True):
    fig = plt.figure()
    ax = plt.axes()
    name = yf.Industry(yf.Ticker(stock_list[0]).info.get('industryKey'))
    ax.set_title(name)
    multi_norm_price = pd.DataFrame()

    for stock in stock_list:    
        price = yf.download([stock], period="2y", auto_adjust = True, progress=False)["Close"]
        if(display_all):     
            ax.plot(price_normalization(price), '--')
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()
    ax.plot(agg_norm_price, color="black")
    
    plt.tight_layout()
    plt.show()