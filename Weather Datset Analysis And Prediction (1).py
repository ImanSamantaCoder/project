#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential


# In[2]:


df=pd.read_csv(r"D:\ALL DATASET\weathernew.csv")


# In[3]:


new_df=df


# In[4]:


#new_df['Date']=df['Formatted Date'].apply(lambda x:x.split()[0])
new_df['DATE AND TIME']=df['Formatted Date'].apply(lambda x:x.split()[0])+" "+df['Formatted Date'].apply(lambda x:x.split()[1].split(".")[0])
new_df['DATE AND TIME']=pd.to_datetime(new_df['DATE AND TIME'])


# In[5]:


new_df=new_df.drop(columns=['Formatted Date','Loud Cover','Daily Summary'])


# In[6]:


new_df.info()


# In[7]:


new_df.shape


# In[8]:


new_df['Summary'].value_counts()


# In[9]:


new_df= new_df[(new_df["Summary"] == "Clear") | (new_df["Summary"] == "Overcast")|(new_df["Summary"] == "Foggy")]


# In[10]:


#new_df['Summary']=new_df['Summary'].replace({'Partly Cloudy':4})
new_df['Summary'].value_counts()


# In[11]:


new_df['Summary'].value_counts()


# In[12]:


new_df.isnull().sum()


# In[13]:


new_df.dropna(inplace=True)


# In[14]:


new_df


# In[15]:


new_df.isnull().sum()


# In[16]:


new_df['Year']=new_df['DATE AND TIME'].dt.year
new_df['Month']=new_df['DATE AND TIME'].dt.month_name()
new_df['Day']=new_df['DATE AND TIME'].dt.day
new_df['day_name']=new_df['DATE AND TIME'].dt.day_name()
new_df['iso_week_number'] = new_df['DATE AND TIME'].dt.isocalendar().week
new_df['Hour']=new_df['DATE AND TIME'].dt.hour


# In[17]:


new_df


# In[18]:


new_df=new_df.drop(columns=['DATE AND TIME','day_name'])


# In[19]:


new_df


# In[20]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
new_df['Month']=lb.fit_transform(new_df['Month'])
new_df.info()


# In[21]:


new_df['Summary'].value_counts()


# In[22]:


new_df['Summary']=new_df['Summary'].replace({'Clear':1,'Overcast':0,'Foggy':2})


# In[23]:


new_df['Summary'].value_counts()


# In[24]:


new_df.info()


# In[25]:


new_df['Precip Type']=lb.fit_transform(new_df['Precip Type'])
new_df.info()


# In[ ]:





# In[26]:


new_df.info()


# In[27]:


new_df.sample(5)


# In[28]:


plt.boxplot(new_df['Temperature (C)'])


# In[29]:


percentile25=new_df['Temperature (C)'].quantile(.25)
percentile75=new_df['Temperature (C)'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Temperature (C)']=np.where(
new_df['Temperature (C)']>upper_limit,
upper_limit,
np.where(
new_df['Temperature (C)']<lower_limit,
lower_limit,
new_df['Temperature (C)']))


# In[30]:


#outlier removed finally
plt.boxplot(new_df['Temperature (C)'])


# In[31]:


plt.boxplot(new_df['Humidity'])


# In[32]:


percentile25=new_df['Humidity'].quantile(.25)
percentile75=new_df['Humidity'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Humidity']=np.where(
new_df['Humidity']>upper_limit,
upper_limit,
np.where(
new_df['Humidity']<lower_limit,
lower_limit,
new_df['Humidity']))


# In[33]:


plt.boxplot(new_df['Humidity'])


# In[34]:


plt.boxplot(new_df['Apparent Temperature (C)'])


# In[35]:


percentile25=new_df['Apparent Temperature (C)'].quantile(.25)
percentile75=new_df['Apparent Temperature (C)'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Apparent Temperature (C)']=np.where(
new_df['Apparent Temperature (C)']>upper_limit,
upper_limit,
np.where(
new_df['Apparent Temperature (C)']<lower_limit,
lower_limit,
new_df['Apparent Temperature (C)']))


# In[36]:


#outlier removed finally
plt.boxplot(new_df['Apparent Temperature (C)'])


# In[37]:


plt.boxplot(new_df['Wind Bearing (degrees)'])


# In[38]:


plt.boxplot(new_df['Visibility (km)'])


# In[39]:


plt.scatter(new_df['Temperature (C)'],new_df['Humidity'])
plt.xlabel("Temperature")
plt.ylabel("humidity")


# In[40]:


plt.figure(figsize=(500,400))
cat_plot=sns.catplot(data=new_df,x='Month',y='Temperature (C)',kind='bar',col='Year',col_wrap=3,sharex=False)
cat_plot.set_xticklabels(rotation=90)
plt.show()
sns.displot(data=new_df,x='Temperature (C)',kind='kde',col='Year',fill=True,col_wrap=3)
cat_plot.set_xticklabels(rotation=90)
plt.show()


# In[41]:


plt.figure(figsize=(50,40))
cat_plot=sns.catplot(data=new_df,x='Month',y='Humidity',kind='bar',col='Year',col_wrap=3,sharex=False)
cat_plot.set_xticklabels(rotation=90)
plt.show()


# In[42]:


plt.figure(figsize=(15,18))
ax=plt.subplot(projection='3d')
ax.scatter3D(new_df['Temperature (C)'],new_df['Humidity'],new_df['Visibility (km)'])
ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.set_zlabel('Visitbility')


# In[ ]:





# In[ ]:





# In[43]:


new_df


# In[ ]:





# In[44]:


new_df['Summary'].value_counts()


# In[45]:


sns.heatmap(new_df.corr(),annot=True, cmap='coolwarm', linewidths=.5)


# In[46]:


new_df=new_df.drop(columns=['Day','Hour','iso_week_number'])


# In[47]:


new_df


# In[48]:


x=new_df.iloc[:,1:]
x.shape
y=new_df.iloc[:,0]
x.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


x.shape
y.shape


# In[50]:


#np.mean(cross_val_score(classifier,x,y,cv=5,scoring='accuracy'))


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from tensorflow.keras.utils import to_categorical
y_trainn = pd.get_dummies(y_train)
model=Sequential()
model.add(Dense(10,activation='relu',input_dim=10))
model.add(Dense(10,activation='relu'))

model.add(Dense(3,activation='softmax'))


# In[52]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['Accuracy'])


# In[53]:


model.fit(x_train,y_trainn,epochs=200)


# In[82]:


y_pred=model.predict(x_test)
y_pred1=y_pred.argmax(axis=1)


# In[66]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)


# In[ ]:




