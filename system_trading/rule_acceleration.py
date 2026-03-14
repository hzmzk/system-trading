from backtest import *
from util import volatility, volatility_list, common_index

def accel(price, fast_span):
    match fast_span:
        case 8:
            ewmac_forecast_scalar = 6.65208
            accel_forecast_scalar = 1.4799
        case 16:
            ewmac_forecast_scalar = 4.64402
            accel_forecast_scalar = 1.46374
        case _:
            ewmac_forecast_scalar = 0
            accel_forecast_scalar = 0

    slow_span = fast_span * 4
    raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()
    
    price_volatility = price.iloc[-1].item() * volatility(price)

    ewmac_forecast = raw_difference / price_volatility * ewmac_forecast_scalar
    forecast = ( ewmac_forecast - ewmac_forecast.shift(fast_span) ).iloc[-1].item() * accel_forecast_scalar
    forecast = min(max(forecast,-20), 20) 
 
    return forecast

def accel_list(price, fast_span):
    match fast_span:
        case 8:
            ewmac_forecast_scalar = 6.65208
            accel_forecast_scalar = 1.4799
        case 16:
            ewmac_forecast_scalar = 4.64402
            accel_forecast_scalar = 1.46374
        case _:
            ewmac_forecast_scalar = 0
            accel_forecast_scalar = 0

    slow_span = fast_span * 4
    raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()

    price_volatility = price * volatility_list(price)

    ewmac_forecast = raw_difference / price_volatility * ewmac_forecast_scalar

    forecast = ( ewmac_forecast - ewmac_forecast.shift(fast_span) ) * accel_forecast_scalar

    column_name = price.columns.item()
    forecast[forecast[column_name] > 20] = 20
    forecast[forecast[column_name] < -20] = -20
    forecast = forecast[slow_span:]
        
    return forecast

def multi_accel(price):
    accel8  = accel_list(price, 8)
    accel16 = accel_list(price, 16)

    forecast = ( accel8.iloc[-1].item() + accel16.iloc[-1].item() ) / 2
    return forecast

def multi_accel_list(price):
    accel8  = accel_list(price, 8)
    accel16 = accel_list(price, 16)

    index = common_index([accel8, accel16])
    forecast = ( accel8.loc[index] + accel16.loc[index] ) / 2

    return forecast

#######################################################################################################

def test_accel(price, parameter, statistics):
    forecast = accel_list(price, parameter)
    return rule_test(price,forecast,statistics)

def test_multi_accel(statistics, price):
    forecast = multi_accel_list(price)
    return rule_test(price,forecast,statistics)







