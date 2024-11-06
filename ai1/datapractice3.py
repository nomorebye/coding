from my_imports import *

#데이터 불러오기 (2013~2023 훈련데이터, 2024 01~04 테스트데이터)
train_data = yf.download('AAPL', start='2013-01-01', end='2024-01-01')
test_data = yf.download('AAPL', start='2024-01-01', end='2024-05-01')
print(f"{train_data}")
print(f"{test_data}")

#데이터 전처리 , 피처 엔지니어링
for data in [train_data, test_data]:
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'].squeeze()).rsi()
    #dropna() 전에 데이터 크기 확인
    print(f"Before dropna: {data.shape}")
    data.dropna(inplace=True)
    print(f"After dropna: {data.shape}")

#피처와 타깃 변수 설정
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI']

X_train = train_data[features].values
y_train = (train_data['Close']).shift(-1) > train_data['Close'].astype(int)
X_train = X_train[:-1]

X_test = test_data[features].values
y_test = (test_data['Close']).shift(-1) > test_data['Close'].astype(int)
X_test = X_test[:-1]

#데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
