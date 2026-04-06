from backtest import *

def breakout(price, horizon):
    match horizon:
        case 20:
            forecast_scalar = 0.88501
        case 40:
            forecast_scalar = 0.928
        case 80:
            forecast_scalar = 0.95375
        case 160:
            forecast_scalar = 0.96842
        case _:
            forecast_scalar = 0

    upper_range = price.rolling(horizon).max()
    lower_range = price.rolling(horizon).min()

    mean = ( upper_range + lower_range ) / 2
   
    forecast = ( 40 * (price - mean) / (upper_range - lower_range) ).ewm(span=(horizon/4), adjust=False).mean().iloc[-1].item() * forecast_scalar

    forecast = min(max(forecast,-20), 20) 
    
    return forecast

def breakout_list(price, horizon):
    match horizon:
        case 20:
            forecast_scalar = 0.88501
        case 40:
            forecast_scalar = 0.928
        case 80:
            forecast_scalar = 0.95375
        case 160:
            forecast_scalar = 0.96842
        case _:
            forecast_scalar = 0

    upper_range = price.rolling(horizon).max()
    lower_range = price.rolling(horizon).min()

    mean = ( upper_range + lower_range ) / 2
    
    forecast = ( 40 * (price - mean) / (upper_range - lower_range) ).ewm(span=(horizon/4), adjust=False).mean() * forecast_scalar
    
    column_name = price.columns.item()
    forecast[forecast[column_name] > 20] = 20
    forecast[forecast[column_name] < -20] = -20
    forecast = forecast[horizon:]
    
    return forecast

def multi_breakout(price):
    breakout20  = breakout_list(price, 20)
    breakout40  = breakout_list(price, 40)
    breakout80  = breakout_list(price, 80)

    forecast = ( breakout20.iloc[-1].item() + breakout40.iloc[-1].item() + breakout80.iloc[-1].item() ) / 3

    return forecast

def multi_breakout_list(price):
    breakout20  = breakout_list(price, 20)
    breakout40  = breakout_list(price, 40)
    breakout80  = breakout_list(price, 80)

    index = common_index([breakout20, breakout40, breakout80])
    forecast = ( breakout20.loc[index] + breakout40.loc[index] + breakout80.loc[index] ) / 3

    return forecast

#######################################################################################################

def test_breakout(price, parameter, statistics):
    forecast = breakout_list(price, parameter)
    return rule_test(price,forecast,statistics)

def test_multi_breakout(statistics, price):
    forecast = multi_breakout_list(price)
    return rule_test(price,forecast,statistics)

