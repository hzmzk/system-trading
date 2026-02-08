import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import volatility

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
    return (rule_outcome * detrended_daily_return.loc[rule_outcome.index])[column_name]

def cap_forecast(forecast):
    column_name = forecast.columns.item()
    forecast[(forecast[column_name] < 5) & (forecast[column_name] > -5)] = 0
    forecast[forecast[column_name] > 5] = 1
    forecast[forecast[column_name] < -5] = -1
    return forecast

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

def get_sharpe_ratio(forecast, price):
    daily_return = price/price.shift() - 1
    daily_return = daily_return[1:]

    common_index = forecast.index.intersection(price.index)

    beginning = common_index.index[:1].item()
    final = common_index.index[-1:].item()

    risk_free_rate = irx_risk_free_rate(start_date=beginning, end_date=final)

    rule_excess_return = (forecast[common_index] * daily_return[common_index] - risk_free_rate[common_index]).mean()

    rule_excess_return_std = (forecast[common_index] * daily_return[common_index] - risk_free_rate[common_index]).std()

    sharpe_ratio = ( rule_excess_return / rule_excess_return_std ) * 16

    return sharpe_ratio 