import pandas as pd

from datetime import datetime

from ib_async import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

def create_stock_data(ticker):
    
    contract = Stock(ticker, "SMART", "USD")

    max_period = str(datetime.now().year - ib.reqHeadTimeStamp(contract, 'ADJUSTED_LAST', True).year + 1) + ' Y'

    bar = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=max_period,
        barSizeSetting="1 day",
        whatToShow="ADJUSTED_LAST",
        useRTH=True,
        formatDate=1
    )

    df = util.df(bar)
    
    datetime_index = pd.DatetimeIndex(df["date"], name="Date")

    df = df.drop("date", axis="columns")

    parameter = ["Open", "High", "Low", "Close", "Volume", "Average", "BarCount"]

    parameter_column = pd.MultiIndex.from_product([parameter,[ticker]], names=['Price','Ticker'])

    data = df[df.columns].apply(pd.to_numeric).to_numpy()

    df = pd.DataFrame(data, index=datetime_index, columns=parameter_column)
 
    df.to_csv('ib_price_data/' + ticker + '.csv')
