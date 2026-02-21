import pandas as pd

def volatility(price):
    daily_return = price / price.shift() - 1
    ewma_mean = daily_return.ewm(span=36, adjust=False).mean().iloc[-1].item()
    ewma_vol = (((daily_return - ewma_mean)**2).ewm(span=36, adjust=False).mean() ** 0.5).iloc[-1].item()
    return ewma_vol


def partition_list(mylist, partition=8):
    list_length = len(mylist)
    partitioned_list = [mylist[ i * list_length // partition : (i+1) * list_length // partition ] for i in range(partition)]
    
    return partitioned_list 


def datetime_csv(file_name, start="", end=""):
    df = pd.read_csv(file_name)
    datetime_index = pd.DatetimeIndex(df["Date"])
    df = df.drop("Date", axis="columns")
    df = df.set_index(datetime_index)

    if(start == ""):
        start = df.index[0]
    if(end == ""):
        end = df.index[-1]

    return df[start:end] 


def common_index(df_list):
    common_index = df_list[0].index
    df_list = df_list[1:]
    for df in df_list:
        common_index = common_index.intersection(df.index)
    return common_index


def inertia(forecast, minimum_change=0.01):
    new_forecast = forecast.copy()

    for i in range(len(forecast.index) - 1):
        percent_change = forecast.iloc[i+1].item() / new_forecast.iloc[i].item() - 1
       
        if(abs(percent_change) <= minimum_change):
            new_forecast.iloc[i+1] = new_forecast.iloc[i]
        else:
            new_forecast.iloc[i+1] = forecast.iloc[i+1]
    return new_forecast


def price_normalization(price, truncate=100):
    std_percentage = pd.DataFrame(columns=price.columns, index=price.index)
    for tally in range(len(price.index)):
        std_percentage.iloc[tally] = volatility(price.iloc[:tally + 1])

    price = price.iloc[truncate:]
    std_percentage = std_percentage.iloc[truncate:]
    
    normalized_daily_return = (price - price.shift()) / (std_percentage * price)
    stock_name = price.columns[0]
    normalized_daily_return.loc[normalized_daily_return[stock_name] > 6] = 6
    normalized_daily_return.loc[normalized_daily_return[stock_name] < -6] = -6
    
    normalized_price = normalized_daily_return.cumsum()
    return normalized_price
   

def correlation_heatmap(multi_price, show_label=False):
    arr   = multi_price.corr().to_numpy().round(3)
    size  = arr.shape[0]
    names = multi_price.corr().index.to_list()

    fig, ax = plt.subplots()
    im      = ax.imshow(arr, cmap="RdBu", vmin=-1, vmax=1)

    if(show_label):
        ax.set_xticks(range(size), labels=names)
        ax.set_yticks(range(size), labels=names)

    for tally1 in range(size):
        for tally2 in range(size):
            text = ax.text(tally2, tally1, arr[tally1,tally2], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()


