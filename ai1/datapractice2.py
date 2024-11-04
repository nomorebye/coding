from my_imports import *
from my_functions import moving_average, volume_moving_average, relative_strength_index, get_confusion_matrix

# 기술 지표 적용
df = moving_average(df, 45)
df = volume_moving_average(df, 45)
df = relative_strength_index(df, 14)

# Dates 열을 인덱스로 설정
df = df.set_index('Dates')
df = df.dropna()
print(len(df))

#타깃 변수 설정
df['pct_change'] = df['CLOSE_SPY'].pct_change()

#모델링을 위한 이진 분륫값 생성
df['target'] = np.where(df['pct_change'] > 0, 1, 0)
df = df.dropna(subset=['target'])

# 정수형 변환
df['target'] = df['target'].astype(np.int64)

print(df['target'].value_counts())

# 다음 날 예측을 위해 타깃 변수를 shift
df['target'] = df['target'].shift(-1)
df = df.dropna()
print(len(df))

# 설명 변수와 타깃 변수 분리
y_var = df['target']
x_var = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)
print(x_var.head())

up = df[df['target'] == 1].target.count()
total = df.target.count()
print('up/down ratio: {0:2f}'. format(up / total))

#훈련 셋과 테스트셋 분할
X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.3, shuffle=False, random_state=3)
train_count = y_train.count()
test_count = y_train.count()

print('train set label ratio')
print(y_train.value_counts() / train_count)
print('test set label ratio')
print(y_test.value_counts() / test_count)

#모델 학습 및 평가
# XGBoost 모델 학습 및 예측
xgb_dis = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_dis.fit(X_train, y_train)
xgb_pred = xgb_dis.predict(X_test)

# 훈련 정확도 확인
print(xgb_dis.score(X_train, y_train))

# 성능 평가
get_confusion_matrix(y_test, xgb_pred)

# 타깃 변수 통제 확인
print(df['pct_change'].describe())