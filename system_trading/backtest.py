import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pickle
import json

from util import volatility, volatility_list, common_index, datetime_csv
from trade_class import Stock
from csv_dataset import industry_basket


###############################################################################################################################

def rule_test(ticker, forecast, statistics, start, end):
    capital = 1000
    transaction_cost = 0.5

    price = Stock(ticker, start_date=start, end_date=end).price

    match statistics:
        case "forecast_average":
            return 10 / abs(forecast).mean()
        

    #forecast = enter_exit_position(forecast) * 10
    position = risk_target_position(price,forecast)

    match statistics:
        case "sharpe_ratio":
            investment_return = strategy_return(price, position)
            return sharpe_ratio(investment_return)
        
        case "post_cost_sharpe_ratio":
            position = position_inertia(position)

            investment_return = strategy_return(price, position)
            
            sr = sharpe_ratio(investment_return)
            turnover_count = turnover(position)
            cost_sr = trade_cost_sr(price, capital, transaction_cost)
            
            return sr - cost_sr * turnover_count
        
        case "turnover":
            position = position_inertia(position)
            return turnover(position)
        
        case "bootstrap":
            return p_value_bootstrap(price, position)
        
        case "monte_carlo":
            return p_value_montecarlo(price, position)
        
        case "win_percentage":
            return win_percent(price, position)
        
        case "test":
            return p_value_bootstrap(price, position)


###############################################################################################################################

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
    daily_return = daily_return.shift(-1)[:-1]

    index = common_index([number_position, daily_return])

    number_position = number_position.loc[index]
    daily_return = daily_return.loc[index]

    return number_position * daily_return


def risk_target_position(price, forecast, annual_volatility_target = 0.2):
    index = common_index([forecast, price])
    forecast = forecast.loc[index]
    price = price.loc[index]

    volatility_target = annual_volatility_target / 16
    percent_allocated = volatility_target / volatility_list(price) * (forecast / 10)
    
    column_name = percent_allocated.columns.item()
    percent_allocated[percent_allocated[column_name] > 1.5 ] = 1.5
    percent_allocated[percent_allocated[column_name] < -1.5 ] = -1.5

    return percent_allocated[2:] 


###############################################################################################################################


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
    detrended_daily_return = detrended_daily_return.shift(-1)[:-1]
    return detrended_daily_return

def rule_return(position, daily_return):
    column_name = position.columns.item()
    index = common_index([position, daily_return])
    position = position.loc[index]
    daily_return = daily_return.loc[index]
    return (position * daily_return)[column_name]

###############################################################################################################################


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

def trade_cost_sr(price, capital, transaction_cost):
    volatility = volatility_list(price)[-252:].mean().item()
    cost_sr = 2 * transaction_cost / (capital * volatility * 16)
    return cost_sr

# def trade_cost_sr(price, transaction_cost):
#     current_price = price.iloc[-1].item()
#     if(current_price < transaction_cost * 1000):
#         block_value = transaction_cost * 1000
#     else:
#         block_value = current_price
#     volatility = volatility_list(price)[-252:].mean()
#     cost_sr = 2 * transaction_cost / (block_value * volatility * 16)
#     return cost_sr
    
def win_percent(price, forecast):
    daily_return = price/price.shift() - 1
    daily_return = daily_return.shift(-1)[:-1]

    column_name = price.columns.item()
    index = common_index([daily_return, forecast])
    outcome = forecast.loc[index] * daily_return.loc[index]

    total = outcome.size
    win_rate = outcome[outcome[column_name] > 0].size / total

    return win_rate

######################################################################################################

def update_rule_performance(rule_name, stats, results):
    if(not rule_name + ".json" in os.listdir("rule_performance/")):
        with open('rule_performance/' + rule_name + '.json', 'w') as f:
            json.dump({}, f)

    with open('rule_performance/' + rule_name + '.json') as f:
        rule_performance = json.load(f)

    with open('rule_performance/' + rule_name + '.json', 'w') as f:
        rule_performance[stats] = results
        json.dump(rule_performance, f)

def stats_summary(rule_list, stats_list):
    for rule in rule_list:
        performance_dict = {}
        with open('rule_performance/' + rule + '.json') as f:
            rule_performance = json.load(f)
        for stats in stats_list:
            value = rule_performance[stats]
            median_value = round(np.median(value),5)
            performance_dict[stats] = median_value
        print(rule)
        for i in performance_dict.keys():
            print(i, " : ", performance_dict[i])   
        print()

######################################################################################################

def generate_jumbo():
    with open('other_data/industry_basket.json') as f:
        industry_company_list = json.load(f).keys()
    mylist = []
    for i in industry_company_list:
        tally = 0
        for j in industry_basket(i):
            price = Stock(j, start_date="2010", end_date="2025").price
            if(price.size > 4000):
                tally = tally + 1
                mylist.append(j)
                if(tally == 2):
                    break
    with open('other_data/jumbo_ticker.json','w') as f:
        json.dump(mylist,f)


def jumbo_ticker_list():
    with open('other_data/jumbo_ticker.json') as f:
        return json.load(f)






