import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from util import volatility, common_index, datetime_csv
from trade_class import Stock

def bootstrap_reality_check(sample):
    zero_centered_sample = sample - np.mean(sample)
    
    bootstrap_distribution = []
    
    for tally in range(2000):
        resample = np.random.choice(zero_centered_sample, len(zero_centered_sample))
        resample_mean = np.mean(resample)
        bootstrap_distribution.append(resample_mean)
    
    return bootstrap_distribution
 
def monte_carlo_permutation(rule_outcome, daily_return):
    rule_outcome = rule_outcome.reset_index(drop=True)
    
    monte_carlo_distribution = []

    for tally in range(2000):
        shuffled_daily_return = daily_return.sample(frac=1).reset_index(drop=True)
        noise_return = rule_outcome * shuffled_daily_return
        monte_carlo_distribution.append(noise_return.mean().item())

    return monte_carlo_distribution
 
def percentile_rank(distribution, value):
    elements_lower_than_value = len([i for i in distribution if i < value]) 
    elements_total = len(distribution)
    return elements_lower_than_value / elements_total

def detrend_return(price):
    daily_return = price/price.shift() - 1
    detrended_daily_return = daily_return - daily_return.mean() 
    detrended_daily_return = detrended_daily_return[1:]
    return detrended_daily_return

def rule_return(rule_outcome, detrended_daily_return):
    column_name = rule_outcome.columns.item()
    index = common_index([rule_outcome, detrended_daily_return])
    return (rule_outcome.loc[index] * detrended_daily_return.loc[index])[column_name]

def cap_forecast(forecast, value_cap=10):
    new_forecast = forecast.copy()
    column_name = forecast.columns.item()
    new_forecast[forecast[column_name] > value_cap] = 10
    new_forecast[forecast[column_name] < -value_cap] = -10
    new_forecast = new_forecast / 10
    return new_forecast

def buffer_forecast(forecast, minimum_change=0.1):
    column_name = forecast.columns.item()
    forecast[forecast[column_name] == 0] = sys.float_info.min
    new_forecast = forecast.copy()
    for i in range(len(forecast.index) - 1):
        abs_percent_change = abs(forecast.iloc[i+1].item() / new_forecast.iloc[i].item() - 1)
        if(abs_percent_change <= minimum_change):
            new_forecast.iloc[i+1] = new_forecast.iloc[i]
        else:
            new_forecast.iloc[i+1] = forecast.iloc[i+1]
    return new_forecast

def turnover(forecast, minimum_change=0.1):
    column_name = forecast.columns.item()
    forecast[forecast[column_name] == 0] = sys.float_info.min
    new_forecast = forecast.copy()
    turnover_count = 0
    for i in range(len(forecast.index) - 1):
        abs_percent_change = abs(forecast.iloc[i+1].item() / new_forecast.iloc[i].item() - 1)
        if(abs_percent_change <= minimum_change):
            new_forecast.iloc[i+1] = new_forecast.iloc[i]
        else:
            new_forecast.iloc[i+1] = forecast.iloc[i+1]
            turnover_count = turnover_count + 1
    total_year = new_forecast.size / 252
    return turnover_count / total_year

def irx_risk_free_rate(start_date, end_date):
    annual_rate = datetime_csv("other_data/^IRX.csv", start=start_date, end=end_date) / 100
     
    # de-annualize
    daily_rate = ( 1 + annual_rate ) ** (1/252) - 1

    daily_rate.columns = ["daily_rf"] 
    return daily_rate  

def sharpe_ratio(price, rule_outcome):
    daily_return = price/price.shift() - 1
    daily_return = daily_return[1:]

    beginning = rule_outcome.index[0]
    final = rule_outcome.index[-1]
    risk_free_rate = irx_risk_free_rate(start_date=beginning, end_date=final)

    index = common_index([rule_outcome, daily_return, risk_free_rate])

    rule_outcome = rule_outcome.loc[index]
    daily_return = daily_return.loc[index]
    risk_free_rate = risk_free_rate.loc[index]

    stock_name = daily_return.columns[0]

    rule_excess_return = ((rule_outcome * daily_return)[stock_name] - risk_free_rate["daily_rf"]).mean()
    rule_excess_return_std = ((rule_outcome * daily_return)[stock_name] - risk_free_rate["daily_rf"]).std()

    sharpe_ratio = ( rule_excess_return / rule_excess_return_std ) * 16
    sharpe_ratio = sharpe_ratio.item()

    return sharpe_ratio 

def p_value_bootstrap(price, rule_outcome):
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(rule_outcome,detrended_daily_return)
    boot_distribution = bootstrap_reality_check(sample_return)
    p_value = 1 - percentile_rank(boot_distribution, sample_return.mean())
    return p_value

def p_value_montecarlo(price, rule_outcome):
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(rule_outcome,detrended_daily_return)
    carlo_distribution = monte_carlo_permutation(rule_outcome, detrended_daily_return)
    p_value = 1 - percentile_rank(carlo_distribution, sample_return.mean())
    return p_value

def rule_stats_summary(price, rule_outcome):
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(rule_outcome,detrended_daily_return)

    boot = bootstrap_reality_check(sample_return)
    carlo = monte_carlo_permutation(rule_outcome, detrended_daily_return)

    print("Mean        : ", "%.5f" % sample_return.mean())

    print("p-value")
    print("Bootstrap   : ", "%.5f" % (1 - percentile_rank(boot, sample_return.mean())))
    print("MonteCarlo  : ", "%.5f" % (1 - percentile_rank(carlo, sample_return.mean())))
    
    plt.hist(boot, label="bootstrap")
    plt.hist(carlo, label="montecarlo")
    plt.legend()
    plt.show()

def price_filter(stock_list, minimum_data=4444):
    price_list = []
    for stock in stock_list:
        price = Stock(stock).price
        if(price.index.size >= minimum_data):
            price_list.append(price[-minimum_data:])
    return price_list

def stock_filter(stock_list, minimum_data=4444):
    new_stock_list = []
    for stock in stock_list:
        price = Stock(stock).price
        if(price.index.size >= minimum_data):
            new_stock_list.append(stock)
    return new_stock_list

def trade_cost(asset_type, price):
    if(asset_type == "stock" or asset_type == "etf"):
        cost = 0.35 
    #price_volatility = price.iloc[-1].item() * volatility(price)
    price_volatility = 2000 * volatility(price)
    cost_sr = 2 * cost / (price_volatility * 16)
    return cost_sr
    
def win_percent(price, forecast):
    daily_return = price/price.shift() - 1
    daily_return = daily_return[1:]

    column_name = price.columns.item()
    index = common_index([daily_return, forecast])
    outcome = forecast.loc[index] * daily_return.loc[index]

    total = outcome.size
    win_rate = outcome[outcome[column_name] > 0].size / total

    return win_rate









