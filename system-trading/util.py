import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt

from system_trading.trading_rule import multi_ewmac


def datetime_csv(file_name):
    df = pd.read_csv(file_name)
    datetime_index = pd.DatetimeIndex(df["Date"])
    df = df.drop("Date", axis="columns")
    df = df.set_index(datetime_index)
    return df


def volatility(price):
    daily_return = price / price.shift() - 1
    ewma_mean = daily_return.ewm(span=36, adjust=False).mean().iloc[-1].item()
    ewma_vol = (((daily_return - ewma_mean)**2).ewm(span=36, adjust=False).mean() ** 0.5).iloc[-1].item()
    return ewma_vol


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


def aggregate_momentum(group, group_key, horizon="15mo"):
    if(group == "sector"):
        stock_list = yf.Sector(group_key).top_companies.index
    elif(group == "industry"):
        stock_list = yf.Industry(group_key).top_companies.index

    multi_norm_price = pd.DataFrame()

    for stock in stock_list:    
        price = yf.download([stock], period=horizon, auto_adjust = True, progress=False)["Close"]
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    
    agg_momentum_forecast = multi_ewmac(agg_norm_price, parameter="normal")

    return agg_momentum_forecast
        

def sector_trend(momentum_value, show_graph = False):
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    passed_sector_list = []
    passed_industry_list = []
    for sector in sector_list:
        sector_print_trigger = True
        multi_industry_norm_price = datetime_csv("dataset/" + sector + ".csv")
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


def industry_aggregate_momentum(industry_key):
    sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    for sector in sector_list:
        multi_industry_norm_price = datetime_csv("dataset/" + sector + ".csv")
        for industry in multi_industry_norm_price.columns:
            if(industry == industry_key):
                industry_aggregate_momentum = multi_ewmac(multi_industry_norm_price[industry], parameter="normal")
                return industry_aggregate_momentum


def plot_agg_norm(stock_list, display_all=True):
    fig = plt.figure()
    ax = plt.axes()
    name = yf.Industry(yf.Ticker(stock_list[0]).info.get('industryKey'))
    ax.set_title(name)
    multi_norm_price = pd.DataFrame()

    for stock in stock_list:    
        price = yf.download([stock], period="1y", auto_adjust = True, progress=False)["Close"]
        if(display_all):     
            ax.plot(price_normalization(price), '--')
        normalized_price = price_normalization(price)
        multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

    agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()
    ax.plot(agg_norm_price, color="black")
    
    plt.tight_layout()
    plt.show()


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
