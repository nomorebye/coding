import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 전역 변수로 데이터 저장
_data_ml = None  # 준비된 데이터를 저장
_training_sample = None
_testing_sample = None
_features = None
_features_short = None

def load_data(file_path, start_date="1999-12-31", end_date="2019-01-01", default_file="data_ml.csv"):
    # 고정 파일 불러오기
    data_fixed = pd.read_csv(default_file)
    data_fixed['date'] = pd.to_datetime(data_fixed['date'])
    
    # 추가 파일 불러오기
    data_raw = pd.read_csv(file_path)
    data_raw['date'] = pd.to_datetime(data_raw['date'])
    
    # 두 데이터를 병합 (date 기준으로 병합)
    data_combined = pd.concat([data_fixed, data_raw], ignore_index=True)
    
    # 날짜 필터링
    idx_date = data_combined.index[
        (data_combined['date'] > start_date) & 
        (data_combined['date'] < end_date)
    ].tolist()
    data_ml = data_combined.iloc[idx_date]
    return data_ml

def extract_features(data_ml):
    features = list(data_ml.iloc[:, 3:95].columns)
    features_short = ["Div_Yld", "Eps", "Mkt_Cap_12M_Usd",
                      "Mom_11M_Usd", "Ocf", "Pb", "Vol1Y_Usd"]
    return features, features_short

def calculate_medians(data_ml):
    df_median = data_ml[['date', 'R1M_Usd', 'R12M_Usd']].groupby(
        ['date']).median()
    df_median.rename(
        columns={"R1M_Usd": "R1M_Usd_median", 
                 "R12M_Usd": "R12M_Usd_median"}, 
        inplace=True)
    data_ml = pd.merge(data_ml, df_median, how='left', on=['date'])
    return data_ml

def add_categorical_features(data_ml):
    data_ml['R1M_Usd_C'] = np.where(
        data_ml['R1M_Usd'] > data_ml['R1M_Usd_median'], 1.0, 0.0)
    data_ml['R12M_Usd_C'] = np.where(
        data_ml['R12M_Usd'] > data_ml['R12M_Usd_median'], 1.0, 0.0)
    return data_ml

def split_data(data_ml, separation_date="2014-01-15"):
    idx_train = data_ml.index[data_ml['date'] < separation_date].tolist()
    idx_test = data_ml.index[data_ml['date'] >= separation_date].tolist()
    training_sample = data_ml[data_ml.index.isin(idx_train)]
    testing_sample = data_ml[data_ml.index.isin(idx_test)]
    return training_sample, testing_sample

def prepare_data_pipeline(file_path, start_date="1999-12-31", end_date="2019-01-01", separation_date="2014-01-15", default_file="data_ml.csv"):
    data_ml = load_data(file_path, start_date, end_date, default_file)
    features, features_short = extract_features(data_ml)
    data_ml = calculate_medians(data_ml)
    data_ml = add_categorical_features(data_ml)
    training_sample, testing_sample = split_data(data_ml, separation_date)
    return data_ml, training_sample, testing_sample, features, features_short

# 데이터 저장 및 관리 함수
# 데이터 저장 및 관리 함수
def prepare_and_store_data(file_path, start_date="1999-12-31", end_date="2019-01-01", separation_date="2014-01-15", default_file="data_ml.csv"):
    global _data_ml, _training_sample, _testing_sample, _features, _features_short
    _data_ml, _training_sample, _testing_sample, _features, _features_short = prepare_data_pipeline(
        file_path, start_date, end_date, separation_date, default_file
    )

# 데이터 접근 함수
def get_data_ml():
    return _data_ml

def get_training_sample():
    return _training_sample

def get_testing_sample():
    return _testing_sample

def get_features():
    return _features

def get_features_short():
    return _features_short


def get_data():
    if _data_ml is None:
        raise ValueError("Data has not been loaded yet. Call prepare_and_store_data() first.")
    return _data_ml
