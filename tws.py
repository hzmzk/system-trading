import pandas as pd

from ib_async import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

def create_ib_price(ticker, period):
    
    contract = Stock(ticker, "SMART", "USD")

    ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=True)

    bar = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=period,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    df = util.df(bar)

    datetime_index = pd.DatetimeIndex(df["date"], name="Date")
    
    df = df.drop("date", axis="columns")
    df = df.set_index(datetime_index)

    return df
