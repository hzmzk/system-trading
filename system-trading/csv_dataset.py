import pandas as pd
import yfinance as yf

import util

def update_agg_norm_price_dataset(sector_list, horizon="15mo"):
    if(sector_list == "all"):
        sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "healthcare", "industrials", "real-estate", "technology", "utilities"]
    elif(sector_list == "first-half"):
        sector_list = ["basic-materials", "communication-services", "consumer-cyclical", "consumer-defensive", "energy", "financial-services", "utilities"]
    elif(sector_list == "second-half"):
        sector_list = ["healthcare", "industrials", "real-estate", "technology"]

    for sector in sector_list:
        industry_list = yf.Sector(sector).industries.index
        multi_agg_norm_price = pd.DataFrame()

        for industry in industry_list:
            if(yf.Industry(industry).top_companies is not None):
                stock_list = yf.Industry(industry).top_companies.index

                multi_norm_price = pd.DataFrame()
                for stock in stock_list:    
                    price = yf.download([stock], period=horizon, auto_adjust = True, progress=False)["Close"]
                    normalized_price = util.price_normalization(price)
                    multi_norm_price = multi_norm_price.join(normalized_price, how="outer")

                agg_norm_price = (multi_norm_price - multi_norm_price.shift()).mean(axis="columns").cumsum()    

                agg_norm_price = pd.DataFrame(agg_norm_price, columns=[industry])
                multi_agg_norm_price = multi_agg_norm_price.join(agg_norm_price, how="outer")

        multi_agg_norm_price.to_csv("dataset/" + sector + ".csv")
    print("Done")


