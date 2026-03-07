from backtest import *
from util import volatility, volatility_list, common_index
from trade_class import Stock

def ewmac(price, fast_span, weight="exponential"):
    match fast_span:
        case 4:
            forecast_scalar = 9.48956
        case 8:
            forecast_scalar = 6.65208
        case 16:
            forecast_scalar = 4.64402
        case 32:
            forecast_scalar = 3.23245
        case _:
            forecast_scalar = 0

    slow_span = fast_span * 4
    raw_difference = price.ewm(span=fast_span, adjust=False).mean().iloc[-1].item() - price.ewm(span=slow_span, adjust=False).mean().iloc[-1].item()
    
    if(weight == "exponential"):
        price_volatility = price.iloc[-1].item() * volatility(price)
    elif(weight == "normal"):
        daily_return = price/price.shift() - 1
        price_volatility = price.iloc[-1].item() * daily_return.rolling(25).std().iloc[-1].item()
        
    forecast = raw_difference / price_volatility * forecast_scalar
    forecast = min(max(forecast,-20), 20) 

    return forecast

def ewmac_list(price, fast_span, weight="exponential"):
    match fast_span:
        case 4:
            forecast_scalar = 9.48956
        case 8:
            forecast_scalar = 6.65208
        case 16:
            forecast_scalar = 4.64402
        case 32:
            forecast_scalar = 3.23245
        case _:
            forecast_scalar = 0

    slow_span = fast_span * 4
    raw_difference = price.ewm(span=fast_span, adjust=False).mean() - price.ewm(span=slow_span, adjust=False).mean()

    if(weight == "exponential"):
        price_volatility = price * volatility_list(price)
    elif(weight == "normal"):
        daily_return = price/price.shift() - 1
        price_volatility = price * daily_return.rolling(25).std()

    forecast = raw_difference / price_volatility * forecast_scalar

    column_name = price.columns.item()
    forecast[forecast[column_name] > 20] = 20
    forecast[forecast[column_name] < -20] = -20
    forecast = forecast[slow_span:]

    return forecast

def multi_ewmac(price, parameter="exponential"):
    ewmac4  = ewmac(price, 4, weight=parameter)
    ewmac8  = ewmac(price, 8, weight=parameter)
    ewmac16 = ewmac(price, 16, weight=parameter)
    #ewmac32 = ewmac(price, 32, weight=parameter)
    
    forecast = ( ewmac4 + ewmac8 + ewmac16 ) / 3

    return forecast

def multi_ewmac_list(price, parameter="exponential"):
    ewmac4  = ewmac_list(price, 4, weight=parameter)
    ewmac8  = ewmac_list(price, 8, weight=parameter)
    ewmac16 = ewmac_list(price, 16, weight=parameter)
    #ewmac32 = ewmac(price, 32, weight=parameter)
    
    index = common_index([ewmac4, ewmac8, ewmac16])
    forecast = ( ewmac4.loc[index] + ewmac8.loc[index] + ewmac16.loc[index] ) / 3

    return forecast

 
#######################################################################################################

def test_ewmac(price, parameter, statistics):
    forecast = ewmac_list(price, parameter)
    return rule_test(price,forecast,statistics)

def test_multi_ewmac(price,statistics):
    forecast = multi_ewmac_list(price)
    return rule_test(price,forecast,statistics)

def ewmac_jumbo_return(price_list, parameter):
    individual_return_list = pd.DataFrame()
    for price in price_list:
        forecast = ewmac_list(price, parameter)
        position = risk_target_position(price, forecast)
        individual_return_list = individual_return_list.join(strategy_return(price, position), how="outer")
    jumbo_return = pd.DataFrame(individual_return_list.mean(axis="columns"),columns=["jumbo_return"])
    return jumbo_return

def multi_ewmac_jumbo_return(price_list):
    individual_return_list = pd.DataFrame()
    for price in price_list:
        forecast = multi_ewmac_list(price)
        position = risk_target_position(price, forecast)
        individual_return_list = individual_return_list.join(strategy_return(price, position), how="outer")
    jumbo_return = pd.DataFrame(individual_return_list.mean(axis="columns"),columns=["jumbo_return"])
    return jumbo_return


