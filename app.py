#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data2020 = pd.read_csv('2020.csv')


# In[3]:


realdata2020 = data2020.iloc[0:366]


# In[4]:


rs2020 = realdata2020['備轉容量(MW)'].to_frame()
rs2020 = rs2020.rename(columns={'備轉容量(MW)':'data'})


# In[5]:


data2021 = pd.read_csv('2021.csv')


# In[6]:


rs2021 = (data2021['備轉容量(萬瓩)'] * 10).to_frame()
rs2021 = rs2021.rename(columns={'備轉容量(萬瓩)':'data'})


# In[7]:


rs = rs2020.append(rs2021)


# In[8]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(rs)


# In[9]:


train_norm = pd.DataFrame(train_norm)


# In[10]:


def generate_data(train):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-37):
        X_train.append(np.array(train.iloc[i:i+30]))
        Y_train.append(np.array(train.iloc[i+30:i+37]))
    return np.array(X_train), np.array(Y_train)
# use last 30 days to predict next 7 days
X_train, Y_train = generate_data(train_norm)


# In[44]:


# def shuffle(X,Y):
#     np.random.seed(10)
#     randomList = np.arange(X.shape[0])
#     np.random.shuffle(randomList)
#     return X[randomList], Y[randomList]
# # shuffle the data, and random seed is 10
# X_train, Y_train = shuffle(X_train, Y_train)


# In[11]:


from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def lstm_stock_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(30,activation='linear'))
    model.add(Dense(7,activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error', 'accuracy'])
    model.summary()
    return model


# In[12]:


model = lstm_stock_model(X_train.shape)


# In[13]:


Y_train = Y_train.reshape([407,7])


# In[62]:


model = lstm_stock_model(X_train.shape)
callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)


# In[63]:


model.save('my_model.h5')


# In[64]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# In[68]:


Predict1 = model.predict(np.array(train_norm[414:444]).reshape((1,30,1)))
print(Predict1)
trainPredict = scaler.inverse_transform(Predict1)
print(trainPredict)


# In[15]:


model = load_model('my_model.h5')


# In[ ]:





# In[16]:


result = model.predict(np.array(train_norm[414:444]).reshape((1,30,1)))
print(result)
real_result = scaler.inverse_transform(result)
print(real_result)


# In[20]:


real_result[0][1]


# In[21]:


f = open('submission.csv', 'w')
print("date,operating_reserve(MW)", file=f)
for i in range(0, 6):
    print(f"2021032{i+3},{int(real_result[0][i])}", file=f)
print(f"20210329,{int(real_result[0][6])}", file=f, end="")


# In[22]:


f.close()


# In[ ]:




