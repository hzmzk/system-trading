import pandas as pd
import yfinance as yf
import numpy as np

from util import volatility, datetime_csv

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
    else:
        raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()


    if(weight == "exponential"):
        price_volatility = price.iloc[-1].item() * volatility(price)
    elif(weight == "normal"):
        price_volatility = price.rolling(25).std().iloc[-1]

    forecast = raw_difference / price_volatility * forecast_scalar

    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
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


def industry_trend_rule(stock, backtest_mode=False):
    ticker = yf.Ticker(stock)
    sector = ticker.info.get('sectorKey')
    industry = ticker.info.get('industryKey')

    multi_industry_norm_price = datetime_csv("industry_normalization_price/" + sector + ".csv", start="2024-01-01")
    industry_norm_price = pd.DataFrame({industry:multi_industry_norm_price[industry]})

    if(not backtest_mode):
        forecast = multi_ewmac(industry_norm_price, parameter="normal")
    else:
        forecast = multi_ewmac(industry_norm_price, parameter="normal", backtest_mode=True)
    
    if(not backtest_mode):
        forecast = min(max(forecast,-20), 20) 
    else:
        column_name = industry_norm_price.columns.item()
        forecast[forecast[column_name] > 20] = 20
        forecast[forecast[column_name] < -20] = -20

    return forecast


def multi_ewmac(price, parameter="exponential", backtest_mode=False):
    ewmac4  = ewmac(price, 4, weight=parameter, backtest_mode=True)
    ewmac8  = ewmac(price, 8, weight=parameter, backtest_mode=True)
    ewmac16 = ewmac(price, 16, weight=parameter, backtest_mode=True)
    #ewmac32 = ewmac(price, 32, weight=parameter)
    
    if(backtest_mode):
        common_index = ewmac8.index
        list_dfs = [ewmac4, ewmac16]
        for df in list_dfs:
            common_index = common_index.intersection(df.index)
        forecast = ( ewmac4.loc[common_index] + ewmac8.loc[common_index] + ewmac16.loc[common_index] ) / 3
    else: 
        forecast = ( ewmac4.iloc[-1].item() + ewmac8.iloc[-1].item() + ewmac16.iloc[-1].item() ) / 3

    return forecast


def multi_accel(price, backtest_mode=False):
    accel8  = acceleration(price, 8, backtest_mode=True)
    accel16 = acceleration(price, 16, backtest_mode=True)

    if(backtest_mode):
        common_index = accel8.index.intersection(accel16.index)
        forecast = ( accel8.loc[common_index] + accel16.loc[common_index] ) / 2
    else: 
        forecast = ( accel8.iloc[-1].item() + accel16.iloc[-1].item() ) / 2

    return forecast


def multi_breakout(price, backtest_mode=False):
    breakout20  = breakout(price, 20, backtest_mode=True)
    breakout40  = breakout(price, 40, backtest_mode=True)
    breakout80  = breakout(price, 80, backtest_mode=True)

    if(backtest_mode):
        common_index = breakout20.index
        list_dfs = [breakout40, breakout80]
        for df in list_dfs:
            common_index = common_index.intersection(df.index)
        forecast = ( breakout20.loc[common_index] + breakout40.loc[common_index] + breakout80.loc[common_index] ) / 3
    else: 
        forecast = ( breakout20.iloc[-1].item() + breakout40.iloc[-1].item() + breakout80.iloc[-1].item() ) / 3

    return forecast
    

def forecast(price):
    individual_forecast = [ multi_ewmac(price) , multi_accel(price) , multi_breakout(price) ]
    weight = [0.3, 0.35, 0.35]
    
    forecast = np.dot(individual_forecast, weight)

    return forecast
 




