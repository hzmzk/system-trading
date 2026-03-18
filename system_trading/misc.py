with open('path_to_file', 'w') as f:
    json.dump(data, f)

with open('path_to_file') as f:
    json.load(f)

f_pickle = open("path_to_file", "wb")
pickle.dump(data, f_pickle)

f_pickle = open("path_to_file", "rb")
pickle.load(f_pickle)

def cap_position(forecast, value_cap=10):
    new_forecast = forecast.copy()
    column_name = forecast.columns.item()
    new_forecast[forecast[column_name] > value_cap] = 10
    new_forecast[forecast[column_name] < -value_cap] = -10
    new_forecast = new_forecast / 10
    return new_forecast

def long_only(price):
    long_list = price.copy()
    column_name = price.columns.item()
    long_list[:] = 10
    return long_list

def cost_embedded_price(price, trading_occurrence, cost_per_share = 0.35):
    index = common_index([price, trading_occurrence])
    price = price.loc[index]
    trading_occurrence = trading_occurrence.loc[index]

    return price + trading_occurrence * cost_per_share

def cost_embedded_position(position, trading_occurrence, cost_per_share = 0.35, capital = 1000):
    index = common_index([position, trading_occurrence])
    position = position.loc[index]
    trading_occurrence = trading_occurrence.loc[index]

    return position + trading_occurrence * cost_per_share / capital

def detrend_return_list(price_list):
    detrended_daily_return_list = []
    for price in price_list:
        detrended_daily_return_list.append(detrend_return(price))
    return detrended_daily_return_list

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

fig = plt.figure()
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');

plt.figure()
plt.subplot(2,1,1)
plt.plot(price)
plt.title("Price")
plt.xticks(rotation=30)

plt.subplot(2,1,2)
plt.plot(price.rolling(25).std())
plt.title("Daily volatility")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

plt.plot(price)
plt.show()

plt.hist(price)
plt.show()


fig = plt.figure()
ax = plt.axes()
ax.plot(price, '--', label="name")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2)
ax[0].plot(price, '--')
ax[1].plot(price)
plt.tight_layout()
plt.show()

price = yf.download(["^GSPC"], period = '1y', auto_adjust = True, progress=False)["Close"]
price = yf.download(["^GSPC"],  start="2024-01-01", end="2025-01-01", auto_adjust = True, progress=False)["Close"]

# 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

# 1m data is only retrievable for the last 7 days
# Anything intraday (interval <1d) only for the last 60 days

price = yf.download(["6262.KL"], period = "1d", interval="30m" , auto_adjust = True, progress=False)["Close"]

#####################################################################################################
match case:
    case "post_cost_sharpe_ratio_price":
        position = risk_target_position_price(price,forecast, capital)
        position = position_inertia(position)
        trade = trade_occurrence(position)
        investment_return = strategy_return_price(price,position, trade, transaction_cost)
        return sharpe_ratio_price(investment_return, capital)

    case "post_cost_sharpe_ratio_percent":
        position = position_inertia(position)
        trade = trade_occurrence(position)
        investment_return = strategy_return(price, position, True, trade, capital, transaction_cost)
        return sharpe_ratio(investment_return)


def sharpe_ratio_price(investment_return, capital):
    beginning = investment_return.index[0]
    final = investment_return.index[-1]
    risk_free_rate = irx_risk_free_rate(start_date=beginning, end_date=final)

    index = common_index([investment_return, risk_free_rate])

    investment_return = investment_return.loc[index]
    risk_free_rate = risk_free_rate.loc[index]

    stock_name = investment_return.columns.item()

    rule_excess_return = (investment_return[stock_name] - capital * risk_free_rate["daily_rf"]).mean()
    rule_excess_return_std = (investment_return[stock_name] - capital * risk_free_rate["daily_rf"]).std()

    sharpe_ratio = ( rule_excess_return / rule_excess_return_std ) * 16
    sharpe_ratio = sharpe_ratio.item()

    return sharpe_ratio

def strategy_return_price(price, number_position, trade_day, trade_cost):
    daily_return = price/price.shift() - 1
    daily_return = daily_return.shift(-1)[:-1]

    index = common_index([number_position, daily_return,price])

    number_position = number_position.loc[index]
    price = price.loc[index]
    daily_return = daily_return.loc[index]

    return price * number_position * daily_return - trade_day * trade_cost

def risk_target_position_price(price, forecast, capital, annual_volatility_target = 0.2):
    index = common_index([forecast, price])
    forecast = forecast.loc[index]
    price = price.loc[index]

    volatility_target = annual_volatility_target / 16
    position = capital * volatility_target / (price * volatility_list(price)) * (forecast / 10)
    
    column_name = position.columns.item()
    for i in range(position.index.size):
        if(position.iloc[i].item() > 1.5 * capital / price.iloc[i].item()):
            position.iloc[i] = 1.5 * capital / price.iloc[i]
        elif(position.iloc[i].item() < -1.5 * capital / price.iloc[i].item()):
            position.iloc[i] = -1.5 * capital / price.iloc[i]
    return position[2:]  

def trade_occurrence(asset_position):
    trade = asset_position.copy()
    trade.iloc[0] = 0

    size = len(asset_position.index) - 1

    for i in range(size):
        difference = (asset_position.iloc[i+1] - asset_position.iloc[i]).item()
        if(abs(difference) > 0):
            trade.iloc[i+1] = 1
        else:
            trade.iloc[i+1] = 0
            
    return trade

def strategy_return(price, number_position, include_cost=False, trade_occurence=0, capital=0, transaction_cost=0):
    daily_return = price/price.shift() - 1
    daily_return = daily_return.shift(-1)[:-1]

    index = common_index([number_position, daily_return])

    number_position = number_position.loc[index]
    daily_return = daily_return.loc[index]

    if(not include_cost):
        return number_position * daily_return
    else:
        return number_position * daily_return - trade_occurence * transaction_cost / capital