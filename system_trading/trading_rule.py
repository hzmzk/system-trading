import pandas as pd

from util import volatility

def ewmac(price, fast_span, weight="exponential", backtest_mode=False):
    match fast_span:
        case 4:
            forecast_scalar = 8.53
        case 8:
            forecast_scalar = 5.95
        case 16:
            forecast_scalar = 4.1
        case 32:
            forecast_scalar = 2.79
        case _:
            forecast_scalar = 0

    slow_span = fast_span * 4

    if(not backtest_mode):
        raw_difference = price.ewm(span=fast_span, adjust=False).mean().iloc[-1].item() - price.ewm(span=slow_span, adjust=False).mean().iloc[-1].item()
    elif(backtest_mode):
        raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()


    if(weight == "exponential"):
        price_volatility = price.iloc[-1].item() * volatility(price)
    elif(weight == "normal"):
        price_volatility = price.rolling(25).std().iloc[-1]

    forecast = raw_difference / price_volatility * forecast_scalar

    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    elif(backtest_mode):
        column_name = price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20
        forecast = forecast[slow_span:]

    return forecast

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
    elif(backtest_mode):
        forecast = ( ewmac_forecast - ewmac_forecast.shift(fast_span) ) * accel_forecast_scalar

    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    elif(backtest_mode):
        column_name = price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20
        forecast = forecast[slow_span:]
        
    return forecast

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
    elif(backtest_mode):
        forecast = ( 40 * (price - mean) / (upper_range - lower_range) ).ewm(span=(horizon/4), adjust=False).mean() * forecast_scalar
    
    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    elif(backtest_mode):
        column_name = price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20
        forecast = forecast[horizon:]
    
    return forecast

def channel_breakout(price, horizon):
    forecast = price.copy()
    ticker = price.columns[0]
    upper_bound = price.rolling(horizon).max()[ticker]
    lower_bound = price.rolling(horizon).min()[ticker]

    forecast[(price[ticker] <  upper_bound) & (price[ticker] > lower_bound)] = float('NaN')
    forecast[price[ticker] ==  upper_bound] = 1
    forecast[price[ticker] ==  lower_bound] = -1
    forecast[:horizon-1] = 0
    forecast = forecast.ffill()
    forecast = forecast.shift()
    
    forecast = forecast[horizon:]
    return forecast

def multi_ewmac(price, parameter="exponential"):
    ewmac4  = ewmac(price, 4, weight=parameter)
    ewmac8  = ewmac(price, 8, weight=parameter)
    ewmac16 = ewmac(price, 16, weight=parameter)
    #ewmac32 = ewmac(price, 32, weight=parameter)
    
    return ( ewmac4 + ewmac8 + ewmac16 ) / 3
    #return ( ewmac4 + ewmac8 + ewmac16 + ewmac32 ) / 4


def multi_accel(price):
    accel8  = acceleration(price, 8)
    accel16 = acceleration(price, 16)
    
    return ( accel8 + accel16 ) / 2

def multi_breakout(price):
    breakout20  = breakout(price, 20)
    breakout40  = breakout(price, 40)
    breakout80  = breakout(price, 80)
    
    return ( breakout20 + breakout40 + breakout80 ) / 3

def forecast(price):
    forecast = multi_ewmac(price) * 0.3 + multi_accel(price) * 0.35 + multi_breakout(price) * 0.35
    forecast = min(max(forecast,-20), 20) 
    
    return forecast
 




