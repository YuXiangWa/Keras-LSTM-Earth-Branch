# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:34:43 2020

@author: user
"""

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

# 設定隨機的種子, 以便每次結果是相同的
numpy.random.seed(7)

# ### STEP2. 準備資料

# 定義數據集
alphabet = "子丑寅卯辰巳午未申酉戌亥"

# 創建字符映射到整數（0 - 11)和反相的查詢字典物件
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

print("地支對應到數字編號: \n", char_to_int)
print("\n")

print("數字編號對應到地支: \n", int_to_char)

# ### STEP3. 準備訓練用資料
# 
# 現在我們需要創建我們的輸入(X)和輸出(y)來訓練我們的神經網路。我們可以通過定義一個輸入序列長度，然後從輸入地支序列中讀取序列。
# 例如，我們使用輸入長度1.從原始輸入數據的開頭開始，我們可以讀取第一個地支“子”，下一個地支作為預測“丑”。我們沿著一個字符移動並重複，直到達到“亥”的預測。

# 準備輸入數據集
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)

# ### STEP4. 資料預處理
# 將NumPy數組重組為LSTM需要的格式，也就是: (samples, time_steps, features)。
# 同時我們將進行資料的歸一化(normalize)來讓資料的值落於0到1之間。並對標籤值進行one-hot的編碼。

# 重塑 X 資料的維度成為 (samples, time_steps, features)
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))

# 歸一化
X = X / float(len(alphabet))

# one-hot 編碼輸出變量
y = np_utils.to_categorical(dataY)

print("X shape: ", X.shape) # (25筆samples, "1"個時間步長, 1個feature)
print("y shape: ", y.shape)

# ### STEP5. 建立模型

# 創建模型
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()

# ### STEP6. 定義訓練並進行訓練

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# ### STEP7. 評估模型準確率

# 評估模型的性能
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

# ### STEP8. 預測結果

for pattern in dataX:
    # 把12個地支一個個拿進模型來預測會出現的地支
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction) # 機率最大的idx
    result = int_to_char[index] # 看看預測出來的是那一個地支
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result) # 打印結果
