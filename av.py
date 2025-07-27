import os 
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
import pandas as pd 

load_dotenv()

ticker = "MSFT"

ts = TimeSeries(key = os.getenv("ALPHA_VANTANGE_API_KEY"))
data, metadata = ts.get_daily(symbol = ticker, outputsize = "full")

data_df = pd.DataFrame(data)
data_df = data_df.transpose()
data_df.rename(columns = {"1. open" : "Open", '2. high' : "High", '3. low' : "Low", '4. close' : "Close", '5. volume' : "Volume"}, inplace = True)
data_df.to_csv(f"{ticker}.csv")