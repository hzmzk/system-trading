import numpy as np
    
def forecast(price):
    individual_forecast = [ multi_ewmac(price) , multi_accel(price) , multi_breakout(price) ]
    weight = [0.3, 0.35, 0.35]
    
    forecast = np.dot(individual_forecast, weight)

    return forecast
 