from util import volatility, common_index

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

def multi_ewmac(price, parameter="exponential", backtest_mode=False):
    ewmac4  = ewmac(price, 4, weight=parameter, backtest_mode=True)
    ewmac8  = ewmac(price, 8, weight=parameter, backtest_mode=True)
    ewmac16 = ewmac(price, 16, weight=parameter, backtest_mode=True)
    #ewmac32 = ewmac(price, 32, weight=parameter)
    
    if(backtest_mode):        
        index = common_index([ewmac4, ewmac8, ewmac16])
        forecast = ( ewmac4.loc[index] + ewmac8.loc[index] + ewmac16.loc[index] ) / 3
    else: 
        forecast = ( ewmac4.iloc[-1].item() + ewmac8.iloc[-1].item() + ewmac16.iloc[-1].item() ) / 3

    return forecast