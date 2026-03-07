import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pickle

from util import volatility, volatility_list, common_index, datetime_csv
from trade_class import Stock

def sharpe_ratio(investment_return):
    beginning = investment_return.index[0]
    final = investment_return.index[-1]
    risk_free_rate = irx_risk_free_rate(start_date=beginning, end_date=final)

    index = common_index([investment_return, risk_free_rate])

    investment_return = investment_return.loc[index]
    risk_free_rate = risk_free_rate.loc[index]

    stock_name = investment_return.columns.item()

    rule_excess_return = (investment_return[stock_name] - risk_free_rate["daily_rf"]).mean()
    rule_excess_return_std = (investment_return[stock_name] - risk_free_rate["daily_rf"]).std()

    sharpe_ratio = ( rule_excess_return / rule_excess_return_std ) * 16
    sharpe_ratio = sharpe_ratio.item()

    return sharpe_ratio

def strategy_return(price, number_position):
    daily_return = price/price.shift() - 1
    daily_return = daily_return[1:]

    index = common_index([number_position, daily_return])

    number_position = number_position.loc[index]
    daily_return = daily_return.loc[index]

    return number_position * daily_return

def enter_exit_position(forecast):
    new_forecast = forecast.copy()
    in_the_market = False
    for i in range(forecast.size):
        forecast_value = forecast.iloc[i].item()
        if(forecast_value >= 0):
            sign = 1
        else:
            sign = -1
        if(not in_the_market):
            if(abs(forecast_value) > 10):
                new_forecast.iloc[i] = 1 * sign
                in_the_market = True
            else:
                new_forecast.iloc[i] = 0
        else:
            if(abs(forecast_value) < 9):
                new_forecast.iloc[i] = 0
                in_the_market = False
            else:
                new_forecast.iloc[i] = new_forecast.iloc[i-1]        
    return new_forecast

def risk_target_position(price, forecast, capital = 1, annual_volatility_target = 0.2):
    index = common_index([forecast, price])
    forecast = forecast.loc[index]
    price = price.loc[index]

    volatility_target = annual_volatility_target / 16
    percent_allocated = volatility_target / volatility_list(price) * (forecast / 10)
    
    column_name = percent_allocated.columns.item()
    percent_allocated[percent_allocated[column_name] > 1.5 ] = 1.5
    percent_allocated[percent_allocated[column_name] < -1.5 ] = -1.5

    return percent_allocated[2:] 

def position_inertia(asset_position, minimum_change=0.1):
    column_name = asset_position.columns.item()
    asset_position[asset_position[column_name] == 0] = sys.float_info.min
    new_asset_position = asset_position.copy()
    for i in range(len(asset_position.index) - 1):
        abs_percent_change = abs(asset_position.iloc[i+1].item() / new_asset_position.iloc[i].item() - 1)
        if(abs_percent_change <= minimum_change):
            new_asset_position.iloc[i+1] = new_asset_position.iloc[i]
        else:
            new_asset_position.iloc[i+1] = asset_position.iloc[i+1]
    return new_asset_position

def turnover(position):
    trade_count = 0
    previous_value = position.iloc[0].item()
    for i in range(position.size - 1):
        current_value = position.iloc[i + 1].item()
        if(current_value != previous_value):
            trade_count = trade_count + 1
            previous_value = current_value
    turnover_count = trade_count / 2
    total_year = position.size / 252
    return turnover_count / total_year 

def cap_position(forecast, value_cap=10):
    new_forecast = forecast.copy()
    column_name = forecast.columns.item()
    new_forecast[forecast[column_name] > value_cap] = 10
    new_forecast[forecast[column_name] < -value_cap] = -10
    new_forecast = new_forecast / 10
    return new_forecast

def irx_risk_free_rate(start_date, end_date):
    annual_rate = datetime_csv("other_data/^IRX.csv", start=start_date, end=end_date) / 100
     
    # de-annualize
    daily_rate = ( 1 + annual_rate ) ** (1/252) - 1

    daily_rate.columns = ["daily_rf"] 
    return daily_rate 

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

def trade_cost_sr(price, cost = 0.5):
    volatility = volatility_list(price)[-252:].mean()
    cost_sr = 2 * cost / (2000 * volatility * 16)
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

def bootstrap_reality_check(sample):
    zero_centered_sample = sample - np.mean(sample)
    bootstrap_distribution = []
    for tally in range(2000):
        resample = np.random.choice(zero_centered_sample, len(zero_centered_sample))
        resample_mean = np.mean(resample)
        bootstrap_distribution.append(resample_mean)
    return bootstrap_distribution

def p_value_bootstrap(price, forecast):
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return)
    boot_distribution = bootstrap_reality_check(sample_return)
    p_value = 1 - percentile_rank(boot_distribution, sample_return.mean())
    return p_value
 
def monte_carlo_permutation(forecast, daily_return):
    index = common_index([daily_return,forecast])
    forecast = forecast.loc[index]
    daily_return = daily_return.loc[index]

    forecast = forecast.reset_index(drop=True)    
    monte_carlo_distribution = []

    for tally in range(2000):
        shuffled_daily_return = daily_return.sample(frac=1).reset_index(drop=True)
        noise_return = forecast * shuffled_daily_return
        monte_carlo_distribution.append(noise_return.mean().item())

    return monte_carlo_distribution

def p_value_montecarlo(price, forecast):
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return)
    carlo_distribution = monte_carlo_permutation(forecast, detrended_daily_return)
    p_value = 1 - percentile_rank(carlo_distribution, sample_return.mean())
    return p_value
 
def percentile_rank(distribution, value):
    elements_lower_than_value = len([i for i in distribution if i < value]) 
    elements_total = len(distribution)
    return elements_lower_than_value / elements_total

def detrend_return(price):
    daily_return = price/price.shift() - 1
    detrended_daily_return = daily_return - daily_return.mean() 
    detrended_daily_return = detrended_daily_return[1:]
    return detrended_daily_return

def rule_return(position, daily_return):
    column_name = position.columns.item()
    index = common_index([position, daily_return])
    position = position.loc[index]
    daily_return = daily_return.loc[index]
    return (position * daily_return)[column_name]

def stats_summary(rule_list, keyvalue_list):
    for rule in rule_list:
        file_list = [ i[:-4] for i in os.listdir("rule_performance/") if i[-(len(rule) + 4):-4] == rule]

        rule_stats = {}
        for file in file_list:
            f_pickle = open("rule_performance/" + file + ".pkl", "rb")
            value = pickle.load(f_pickle)

            stats = file[:-(len(rule)+1)]
            median_value = round(np.median(value),5)

            rule_stats.update({stats:median_value})
            
        rule_stats = dict(sorted(rule_stats.items(), key=lambda item: item[1]))
        print(rule)
        for keyvalue in keyvalue_list:
            print(keyvalue, ":", rule_stats[keyvalue])
        print()

def dummy(*args, **kargs):
    return 0

def jumbo_price_list():
    f_pickle = open("other_data/jumbo_price.pkl", "rb")
    price_list = pickle.load(f_pickle)
    return price_list

def jumbo_stock_list():
    f_pickle = open("other_data/jumbo_stock.pkl", "rb")
    stock_list = pickle.load(f_pickle)
    return stock_list

def rule_test(price, forecast, statistics):
    match statistics:
        case "forecast_average":
            return 10 / abs(forecast).mean()

    #forecast = enter_exit_position(forecast) * 10
    position = risk_target_position(price,forecast)
    #position = position_inertia(position, minimum_change=0.1)

    match statistics:
        case "sharpe_ratio":
            investment_return = strategy_return(price, position)
            return sharpe_ratio(investment_return)
        case "post_cost_sharpe_ratio":
            position = position_inertia(position)
            investment_return = strategy_return(price,position)
            sr = sharpe_ratio(investment_return)
            turnover_count = turnover(position)
            cost_sr = trade_cost_sr(price)
            return sr - cost_sr * turnover_count
        case "turnover":
            position = position_inertia(position, minimum_change=0.1)
            return turnover(position)
        case "cost_sr":
            position = position_inertia(position)
            investment_return = strategy_return(price,position)
            sr = sharpe_ratio(investment_return)
            return trade_cost_sr(price)
        case "bootstrap":
            return p_value_bootstrap(price, position)
        case "monte_carlo":
            return p_value_montecarlo(price, position)
        case "win_percentage":
            return win_percent(price, position)
        case "forecast_average":
            return 10 / abs(forecast).mean()
        case "test":
            investment_return = strategy_return(price, position)
            return sharpe_ratio(investment_return)








