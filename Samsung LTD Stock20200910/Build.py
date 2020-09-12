import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime

#데이터 읽어오기
data = pd.read_csv('datasets/005930.KS.csv')
data.head()
###################데이터 전처리
#결측치 없애줌(Null Value)
data = data.dropna()

#중간 값 처리
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

#윈도우 생성
seq_len = 50 #최근 50일 데이터만 사용-> 내일 예측
sequence_length = seq_len + 1 # 50일로 51을 예측

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])

#데이터 정규화.
normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

#print(x_train.shape, x_test.shape)

#여기서 모델 만들어 주기
# model = Sequential()
#
# model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
#
# model.add(LSTM(64, return_sequences=False))
#
# model.add(Dense(1, activation='linear'))
#
# model.compile(loss='mse', optimizer='adam')

#model.summary()

#모델 만들어 졌고 그 다음에 training과정.

# model.fit(x_train, y_train,
#     validation_data=(x_test, y_test),
#     batch_size=10,
#     epochs=20)

#model.save("stockmodel1.h5")

new_model = load_model("stockmodel1.h5")

pred = new_model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()