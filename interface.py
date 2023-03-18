import joblib
import streamlit as st
import pickle
import pandas as pd
import numpy as np

timestep = 3

loaded_model = joblib.load('seed_word')
file = open('dict', 'rb')
ddict = pickle.load(file)
w2i = ddict["w2i"]
i2w = ddict["i2w"]
recommended = ddict["recommended"]

# Dự báo
def encode(sent):
    # Từ thành số
    return [[w2i[w] for w in sent.split()]]

def predict(s):
    
    L = s.split()
    i = 0
    
    while True:
        pred = loaded_model.predict(encode(" ".join(L[i:i+3])))
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
 