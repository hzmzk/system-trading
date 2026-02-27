from backtest import *
from util import volatility, common_index

def acceleration(price, fast_span, backtest_mode=False):
    match fast_span:
        case 8:
            ewmac_forecast_scalar = 5.95
            accel_forecast_scalar = 1.87
        case 16:
            ewmac_forecast_scalar = 4.1
            accel_forecast_scalar = 1.9
        case _:
            ewmac_forecast_scalar = 0
            accel_forecast_scalar = 0

    slow_span = fast_span * 4

    raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()
    price_volatility = price.iloc[-1].item() * volatility(price)

    ewmac_forecast = raw_difference / price_volatility * ewmac_forecast_scalar

    if(not backtest_mode):
        forecast = ( ewmac_forecast - ewmac_forecast.shift(fast_span) ).iloc[-1].item() * accel_forecast_scalar
    else:
        forecast = ( ewmac_forecast - ewmac_forecast.shift(fast_span) ) * accel_forecast_scalar

    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
        column_name = price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20
        forecast = forecast[slow_span:]
        
    return forecast

def multi_acceleration(price, backtest_mode=False):
    accel8  = acceleration(price, 8, backtest_mode=True)
    accel16 = acceleration(price, 16, backtest_mode=True)

    if(backtest_mode):
        index = common_index([accel8, accel16])
        forecast = ( accel8.loc[index] + accel16.loc[index] ) / 2
    else: 
        forecast = ( accel8.iloc[-1].item() + accel16.iloc[-1].item() ) / 2

    return forecast

#######################################################################################################

def dr_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return).mean().item()
    return sample_return

def sr_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    sr = sharpe_ratio(price, forecast)
    return sr

def bt_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    
    bt = p_value_bootstrap(price, forecast)
    return bt

def mc_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    mc = p_value_montecarlo(price, forecast)
    return mc

def turnover_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return turnover(forecast)

def win_percent_acceleration(price, fast_span):
    forecast = acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return win_percent(price, forecast)

#########################################################################################################

def dr_multi_acceleration(price):
    forecast = multi_acceleration(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    detrended_daily_return = detrend_return(price)
    sample_return = rule_return(forecast,detrended_daily_return).mean().item()
    return sample_return

def sr_multi_acceleration(price):
    forecast = multi_acceleration(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    sr = sharpe_ratio(price, forecast)
    return sr

def bt_multi_acceleration(price):
    forecast = multi_acceleration(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    
    bt = p_value_bootstrap(price, forecast)
    return bt

def mc_multi_acceleration(price):
    forecast = multi_acceleration(price, backtest_mode=True) 
    forecast = cap_forecast(forecast)

    mc = p_value_montecarlo(price, forecast)
    return mc

def turnover_multi_acceleration(price, fast_span):
    forecast = multi_acceleration(price, fast_span, backtest_mode=True) 
    forecast = cap_forecast(forecast)
    return turnover(forcast)



