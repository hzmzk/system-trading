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