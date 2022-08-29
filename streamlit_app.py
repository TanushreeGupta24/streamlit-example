#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[2]:


import pandas_datareader as data


# In[67]:


start='2010-01-01'
end='2022-10-31'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df=data.DataReader(user_input,'yahoo',start,end)
df.head

#Describing data

st.subheader('Data from 2010-2019')
st.write(df.describe())
# In[68]:

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200 MA')
ma100=df.close.rolling(100).mean()
ma200=df.close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)




#splitting data into trainng and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
print(data_training.shape)
print(data_testing.shape)





from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


# In[20]:


data_training_array=scaler.fit_transform(data_training)
data_training_array


# In[21]:


x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)


# In[22]:


from keras.layers import Dense,Dropout, LSTM
import keras
from keras.models import Sequential
from tensorflow.keras.layers import ReLU


# In[23]:


keras. __version__


# In[24]:


model=Sequential()
model.add(LSTM(units=60, activation = 'relu' , return_sequences = True , input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))


# In[25]:


model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))


# In[26]:


model.summary()


# In[27]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[28]:


model.fit(x_train, y_train, epochs = 50)


# In[30]:





# In[34]:





# In[37]:


past_100_days=data_training.tail(100)


# In[38]:


final_df=past_100_days.append(data_testing,ignore_index=True)


# In[39]:


final_df.head()


# In[40]:


input_data=scaler.fit_transform(final_df)
input_data


# In[41]:


input_data.shape


# In[42]:


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[43]:


x_test,y_test=np.array(x_test),np.array(y_test)



# In[44]:


y_predicted=model.predict(x_test)




scaler.scale_


# In[58]:


scale_factor=1/0.00682769
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


# In[59]:

st.subheader('Predicted Vs Original Price')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# In[ ]:




