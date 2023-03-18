import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
import collections
import streamlit as st

# =============================================================================
#  Chuẩn bị dữ liệu
# =============================================================================
f = open("text.txt", "r", encoding="utf-8")
text = f.read()

# Loại bỏ kí tự đặt biệt
def keep_alphanum(string):
    L = []
    
    for i in string.split():
        word=''
        for j in i:
            if j.isalnum():
                word += j.lower()
        L.append(word)
    
    return ' '.join(i for i in L)

text_pre = keep_alphanum(text)
len(text_pre)
data = text_pre.split()
data = list(pd.DataFrame(data)[0])

# Đoạn văn có 457 từ khác nhau
len(np.unique(data))

# Tạo 2 từ điển với các từ duy nhất
def build_dataset(words):
    
    wordID = {}
    
    # Count elements from a string
    # most_common để tạo list [('em', 23), ('anh', 21), ('sẽ', 13),...]
    count = collections.Counter(words).most_common()
    print(count)
    # Tạo từ điển ngược (word -> index)
    for word, freq in count:
        wordID[word] = len(wordID)
    # Tạo từ điển xuôi
    idWord = dict(zip(wordID.values(), wordID.keys()))
    
    return wordID, idWord

w2i, i2w = build_dataset(data)
vocab_size = len(w2i)
timestep = 3

# =============================================================================
# Tạo dữ liệu huấn luyện
# =============================================================================

# Với tập X gồm tập hợp các list 3 từ liên tiếp của tập data
#   (trừ list ba từ cuối)
encoded_data = [w2i[w] for w in data]
X = encoded_data[:-1]
Y = encoded_data[timestep:]

# Tạo mô hình và huấn luyện

train_data = keras.preprocessing.timeseries_dataset_from_array(X, Y, sequence_length=timestep, sampling_rate=1)

model = keras.Sequential()
# Add a LSTM layer with 512 internal units.
model.add(LSTM(512, return_sequences=True,
               input_shape=(timestep, 1)))
model.add(LSTM(512, return_sequences=False))
# Add a Dense layer with 10 units.
model.add(Dense(vocab_size))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, epochs=500)

# Tạo từ gợi ý cho giao diện
i = 0
recommended = []
while True:
    word = ''
    for j in X[i:i+timestep]:
        word += i2w[j] + ' '
    recommended.append(word[:-1])
    if i + timestep == len(X):
        break
    i += 1

# Dự báo
def encode(sent):
    # Từ thành số
    return [[w2i[w] for w in sent.split()]]

def predict(s):
    
    L = s.split()
    i = 0
    
    while True:
        pred = model.predict(encode(" ".join(L[i:i+3])))
        # argmax lấy phần tử lớn nhất của list
        L.append(i2w[np.argmax(pred)])
        if len(L) == 13:
            break
        i += 1
    
    result = " ".join(L[timestep:])
    return result


# Interface
st.title(':blue[App sinh từ sử dụng LSTM]')

option = st.selectbox(
    'Mời nhập 3 từ',
    recommended)

if st.button('Sinh từ'):
    st.write(option + ' '+ ':red[' + predict(option)+ ']')