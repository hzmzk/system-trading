from backtest import *

def breakout(price, horizon, backtest_mode=False):
    match horizon:
        case 20:
            forecast_scalar = 0.791
        case 40:
            forecast_scalar = 0.817
        case 80:
            forecast_scalar = 0.837
        case 160:
            forecast_scalar = 0.841
        case _:
            forecast_scalar = 0

    upper_range = price.rolling(horizon).max()
    lower_range = price.rolling(horizon).min()

    mean = ( upper_range + lower_range ) / 2
    
    if(not backtest_mode):
        forecast = ( 40 * (price - mean) / (upper_range - lower_range) ).ewm(span=(horizon/4), adjust=False).mean().iloc[-1].item() * forecast_scalar
    else:
        forecast = ( 40 * (price - mean) / (upper_range - lower_range) ).ewm(span=(horizon/4), adjust=False).mean() * forecast_scalar
    
    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
        column_name = price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20
        forecast = forecast[horizon:]
    
    return forecast

def multi_breakout(price, backtest_mode=False):
    breakout20  = breakout(price, 20, backtest_mode=True)
    breakout40  = breakout(price, 40, backtest_mode=True)
    breakout80  = breakout(price, 80, backtest_mode=True)

    if(backtest_mode):
        index = common_index([breakout20, breakout40, breakout80])
        forecast = ( breakout20.loc[index] + breakout40.loc[index] + breakout80.loc[index] ) / 3
    else: 
        forecast = ( breakout20.iloc[-1].item() + breakout40.iloc[-1].item() + breakout80.iloc[-1].item() ) / 3

    return forecast

#######################################################################################################

def dr_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return).mean().item()
    return sample_return

def sr_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return sharpe_ratio(price, forecast)

def bt_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return p_value_bootstrap(price, forecast)

def mc_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return p_value_montecarlo(price, forecast)

def turnover_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return turnover(forecast)

def win_percent_breakout(price, horizon):
    forecast = breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return win_percent(price,forecast)

#########################################################################################################

def dr_multi_breakout(price):
    forecast = multi_breakout(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return).mean().item()
    return sample_return

def sr_multi_breakout(price):
    forecast = multi_breakout(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    sr = sharpe_ratio(price, forecast)
    return sr

def bt_multi_breakout(price):
    forecast = multi_breakout(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    
    bt = p_value_bootstrap(price, forecast)
    return bt

def mc_multi_breakout(price):
    forecast = multi_breakout(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    mc = p_value_montecarlo(price, forecast)
    return mc

def turnover_multi_breakout(price, horizon):
    forecast = multi_breakout(price, horizon, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return turnover(forcast)
