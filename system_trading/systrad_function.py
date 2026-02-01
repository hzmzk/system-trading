import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt

from util import volatility, datetime_csv
from trading_rule import multi_ewmac


def price_normalization(price, truncate=100):
    std_percentage = pd.DataFrame(columns=price.columns, index=price.index)
    for tally in range(len(price.index)):
        std_percentage.iloc[tally] = volatility(price.iloc[:tally + 1])

    price = price.iloc[truncate:]
    std_percentage = std_percentage.iloc[truncate:]
    
    normalized_daily_return = (price - price.shift()) / (std_percentage * price)
    stock_name = price.columns[0]
    normalized_daily_return.loc[normalized_daily_return[stock_name] > 6] = 6
    normalized_daily_return.loc[normalized_daily_return[stock_name] < -6] = -6
    
    normalized_price = normalized_daily_return.cumsum()
    return normalized_price


def sector_trend(momentum_value, show_graph = False):
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    passed_sector_list = []
    passed_industry_list = []
    for sector in sector_list:
        sector_print_trigger = True
        multi_industry_norm_price = datetime_csv("sector_normalization_price/" + sector + ".csv")
        for industry in multi_industry_norm_price.columns:
            industry_aggregate_momentum = multi_ewmac(multi_industry_norm_price[industry], parameter="normal")
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
def aggregate_momentum(industry_key, horizon="2y"):
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


# get risk-free rate from ^IRX (3-month US Treasury Bill)
def irx_risk_free_rate():
    annual_rate = yf.download("^IRX", period = "max", auto_adjust = True, progress=False)["Close"] / 100
    
    # de-annualize
    daily_rate = ( 1 + annual_rate ) ** (1/252) - 1

    daily_rate.columns = ["daily_rf %"] 
    return daily_rate * 100    


def correlation_heatmap(multi_price, show_label=False):
    arr   = multi_price.corr().to_numpy().round(3)
    size  = arr.shape[0]
    names = multi_price.corr().index.to_list()

    fig, ax = plt.subplots()
    im      = ax.imshow(arr, cmap="RdBu", vmin=-1, vmax=1)

    if(show_label):
        ax.set_xticks(range(size), labels=names)
        ax.set_yticks(range(size), labels=names)

    for tally1 in range(size):
        for tally2 in range(size):
            text = ax.text(tally2, tally1, arr[tally1,tally2], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()
