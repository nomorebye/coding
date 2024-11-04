# 패키지 설치
#pip install -U finance-datareader

# 패키지 import
import FinanceDataReader as fdr
import pandas as pd
from pandas_datareader import data as pdr
import talib
import yfinance as yf

# KRX 종목 리스트 불러오기
df_krx = fdr.DataReader('KRX')
df_krx = fdr.StockListing('KRX')
print(df_krx.head())

# 삼성전자 데이터 불러오기 
sam_data = fdr.DataReader('005930', start="2019-11-06", end="2024-11-06")

# 삼성전자 SMA 
sam_data['SMA_20'] = talib.SMA(sam_data['Close'], timeperiod=20)

# 삼성전자 RSI
sam_data['RSI_14'] = talib.RSI(sam_data['Close'], timeperiod=14)

print(sam_data['RSI_14'].tail())
