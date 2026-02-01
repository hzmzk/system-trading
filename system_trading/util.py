import pandas as pd

def volatility(price):
    daily_return = price / price.shift() - 1
    ewma_mean = daily_return.ewm(span=36, adjust=False).mean().iloc[-1].item()
    ewma_vol = (((daily_return - ewma_mean)**2).ewm(span=36, adjust=False).mean() ** 0.5).iloc[-1].item()
    return ewma_vol

def datetime_csv(file_name, start="", end=""):
    df = pd.read_csv(file_name)
    datetime_index = pd.DatetimeIndex(df["Date"])
    df = df.drop("Date", axis="columns")
    df = df.set_index(datetime_index)

    if(start == ""):
        start = df.index[:1].item()
    if(end == ""):
        end = df.index[-1:].item()

    return df[start:end]