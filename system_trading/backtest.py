import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import volatility, common_index
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

def cap_forecast(forecast):
    column_name = forecast.columns.item()
    forecast[(forecast[column_name] < 5) & (forecast[column_name] > -5)] = 0
    forecast[forecast[column_name] > 5] = 1
    forecast[forecast[column_name] < -5] = -1
    return forecast

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

def size_check(stock_list, minimum_data=500):
    new_stock_list = []
    for stock in stock_list:
        if(Stock(stock).price.index.size >= minimum_data):
            new_stock_list.append(stock)
    return new_stock_list



from trade_class import Stock
from trading_rule import ewmac

def xxx(stock_symbol):
    price = Stock(stock_symbol).price
    forecast = ewmac(price, 16, backtest_mode=True)
    return sharpe_ratio(price, forecast)