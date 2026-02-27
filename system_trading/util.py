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

def price_normalization(price, truncate=100):
    price_volatility = price.rolling(25).std()

    price = price.iloc[truncate:]
    price_volatility = price_volatility[truncate:]

    normalized_daily_return = (price - price.shift()) / (price_volatility)
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

def to_date(x):
    date = str(x)
    date = '-'.join([date[:4], date[4:6], date[6:]])
    return date


def fama_risk_free_rate(fama_daily_rf_csv):
    daily = pd.read_csv(fama_daily_rf_csv)

    date = [to_date(i) for i in daily["Unnamed: 0"]]
    date_index = pd.DatetimeIndex(date)

    risk_free = pd.DataFrame({"rf":daily["RF"].to_list()}, index=date_index)
    risk_free.to_csv("other_data/" + "FAMA_rf" + ".csv")
