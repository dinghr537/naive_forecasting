#!/usr/bin/env python
# coding: utf-8


file1 = "f1"
file2 = "f2"
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='2020.csv 2021.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    print(args)
    print(type(args))
    files = vars(args)
    file1, file2 = files["training"].split()
    print(file1)
    print(file2)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:

# read 2020 data and filter out data of 2020.1, and rename the data column(block2-4)
data2020 = pd.read_csv(file1)


# In[3]:


realdata2020 = data2020.iloc[0:366]


# In[4]:


rs2020 = realdata2020['備轉容量(MW)'].to_frame()
rs2020 = rs2020.rename(columns={'備轉容量(MW)':'data'})


# In[5]:

# read 2021 data and rename the data column(block5-6)
data2021 = pd.read_csv(file2)


# In[6]:


rs2021 = (data2021['備轉容量(萬瓩)'] * 10).to_frame()
rs2021 = rs2021.rename(columns={'備轉容量(萬瓩)':'data'})


# In[7]:

# merge two years' data(block7)
rs = rs2020.append(rs2021)


# In[8]:


print(rs.shape)


# In[9]:

# Feature Scaling the data to 0~1 (block9)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(rs)


# In[10]:


train_norm = pd.DataFrame(train_norm)


# In[11]:

# use source data to generate training data
def generate_data(train):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-38):
        X_train.append(np.array(train.iloc[i:i+30]))
        Y_train.append(np.array(train.iloc[i+30:i+38]))
    return np.array(X_train), np.array(Y_train)
# use last 30 days to predict next 8 days
X_train, Y_train = generate_data(train_norm)


# In[12]:


# import what we need for building the model (block 12-13)
# 參考網路上lstm模型的構建方法，且暫未對模型各層級進行調整（可能等以後對模型有更深的理解後再進行客製化模型）
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

def lstm_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(30,activation='linear'))
    model.add(Dense(8,activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error', 'accuracy'])
    model.summary()
    return model


# In[13]:


model = lstm_model(X_train.shape)


# In[14]:


print(Y_train.shape)


# In[15]:

# adjust the data shape
Y_train = Y_train.reshape([408,8])


# In[17]:

# train
# 此處將其註解掉，助教如要train模型，請解開註解（line 153-156 && line 162 && line 168-171 && line 183-186）
# model = lstm_model(X_train.shape)
# callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
# history = model.fit(X_train, Y_train, epochs=200, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)


# In[21]:


# model.save('my_model.h5')


# In[18]:


# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.show()


# In[19]:


# print(train_norm.shape)


# In[20]:


# Predict = model.predict(np.array(train_norm[416:446]).reshape((1,30,1)))
# print(Predict)
# trainPredict = scaler.inverse_transform(Predict)
# print(trainPredict)


# In[15]:


model = load_model('my_model.h5')



# In[24]:


result = model.predict(np.array(train_norm[416:446]).reshape((1,30,1)))
print(result)
real_result = scaler.inverse_transform(result)
print(real_result)


# In[25]:


real_result[0][1]


# In[26]:

# save result
f = open('submission.csv', 'w')
print("date,operating_reserve(MW)", file=f)
for i in range(1, 7):
    print(f"2021032{i+2},{int(real_result[0][i])-10}", file=f)
print(f"20210329,{int(real_result[0][7])-10}", file=f, end="")


# In[27]:


f.close()


# In[ ]:




