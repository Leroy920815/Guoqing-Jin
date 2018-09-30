#coding=utf-8
import numpy as np
import time
import pandas as pd
from texttable import Texttable
import re
from datetime import datetime
import test_fun
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plt
 
# print(test_fun.sum(1,2))

# data = pd.read_csv('D:/bike_kaggle/train.csv',header= 0,error_bad_lines= False)
# print (data.head())

# result = df.describe()
# result.to_csv('D:/bike_kaggle/t.csv')
# print (df.describe())

#datetime小时计数、season季节、holiday是否假期、workingday工作日、weather天气、temp华氏温度、atemp、humidity湿度、windspeed风速、
#casual非注册租车人数、registered注册租车人数
# print (df.head())
# print (df.info())

# 查看是否有空值数据
def df_isnull(df):
    if df.isnull().any(axis=0).sum():
        print ('有空值')
    else:
        print ('无空值')

# datetime清洗
def time_clean(df):
    # 将datetime切分为日期和时间
    temp = pd.DatetimeIndex(df['datetime'])
    df['year'] = temp.year
    df['date'] = temp.date
    df['time'] = temp.time
    df['month'] = temp.month

    # 获取一周中的第几天
    weekday = [datetime.date(datetime.strptime(time,'%Y-%M-%d')).isoweekday() for time in df['datetime'].str[:10]]
    df['weekday'] = weekday

    # 获取小时数
    df['hour'] = pd.to_datetime(df.time,format='%H:%M:%S')
    df['hour'] = pd.Index(df['hour']).hour

    # 删除原始数据的时间
    df = df.drop('datetime',axis= 1)
    return df
    print (df[:10])

# 离散特征值哑变量处理
def sep_dummies(df,sep_list):
    df = pd.get_dummies(df,columns=sep_list)
    return df
    # print(df.info())

# 连续特征值标准化处理
def conti_standard(df,conti_list):
    for c in conti_list:
        d = df[c]
        max = d.max()
        min = d.min()
        df[c] = (d-min)/(max-min)
    return df
    # print (df)

# 导入数据并转化为DataFrame，train和test合并作数据预处理
data_train = pd.read_csv('D:/Guoqing-Jin/bike_kaggle/train.csv',header= 0,error_bad_lines= False)
data_test = pd.read_csv('D:/Guoqing-Jin/bike_kaggle/test.csv',header= 0,error_bad_lines= False)

df = pd.DataFrame(data_train).append(pd.DataFrame(data_test))

df_isnull(df)

df = time_clean(df)
# print (df[:10])

conti_list = ['temp','atemp','humidity','windspeed']
conti_standard(df,conti_list)
cols = ['count','year','month','weekday','hour','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered']
df = df.ix[:,cols]
df_train = df.iloc[:10886]
df_test = df.iloc[10886:]

# 计算相关系数，查看指标对结果的影响
correlation = df_train.corr()
influence_order = correlation['count'].sort_values(ascending = False)
influence_order_abs = abs(correlation['count'].sort_values(ascending = False))

# 相关性分析热力图
import seaborn as sns
f,ax = plt.subplots(figsize =(16,16))
cmap = sns.cubehelix_palette(light=1,as_cmap=True)
sns.heatmap(df.corr(),vmax=1,annot=True,center=1,cmap=cmap,linewidths=1,ax=ax)

from collections import Counter
# 各指标的影响
# 季节的影响
x = np.unique(df_train['season'])
y = df_train['count'].groupby(df_train['season']).sum()
plt.figure()
plt.plot(x,y)

# 周几的影响
x = np.unique(df_train['weekday'])
y = df_train['count'].groupby(df_train['weekday']).sum()
plt.figure()
plt.plot(x,y)
# print (x)

# 小时的影响
x = np.unique(df_train['hour'])
y = df_train['count'].groupby(df_train['hour']).sum()
plt.figure()
plt.plot(x,y)

# 假期的影响
x = np.unique(df_train['holiday'])
y = df_train['count'].groupby(df_train['holiday']).sum()
plt.figure()
plt.plot(x,y)

# 天气的影响
x = np.unique(df_train['weather'])
y = df_train['count'].groupby(df_train['weather']).sum()
plt.figure()
plt.plot(x,y)

# 气温的影响
x = np.unique(df_train['temp'])
y = df_train['count'].groupby(df_train['temp']).sum()
plt.figure()
plt.plot(x,y)

# 气温的影响
x = np.unique(df_train['windspeed'])
y = df_train['count'].groupby(df_train['windspeed']).sum()
plt.figure()
plt.plot(x,y)

# 湿度的影响
x = np.unique(df_train['humidity'])
y = df_train['count'].groupby(df_train['humidity']).sum()
plt.figure()
plt.plot(x,y)
# plt.show()

# 特征工程
#  删除不要的字段
df = df.drop(['season','atemp','casual','registered'],axis=1)
# 离散变量作one-hot处理
sep_list= ['year','month','weekday','hour','weather']
df_train = sep_dummies(df_train,sep_list)
df_test = sep_dummies(df_test,sep_list)

# 构建模型
# 1、特征向量化
from sklearn.feature_extraction import DictVectorizer
col_trans = ['holiday','workingday','temp','humidity','windspeed','year_2011','year_2012','month_1','month_2','month_3','month_4','month_5','month_6','month_7', 'month_8','month_9','month_10', 'month_11','month_12','hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7','hour_8','hour_9','hour_10','hour_11', 'hour_12','hour_13','hour_14','hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21','hour_22','hour_23','weather_1','weather_2','weather_3','weather_4']
X_train = df_train[col_trans]
X_test = df_test[col_trans]
Y_train = df_train['count']
vec = DictVectorizer(sparse=False)  # spare = False 是不产生稀疏矩阵
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.fit_transform(X_test.to_dict(orient='record'))

# 分割训练数据
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.25,random_state=40)

# 2、建模预测
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# (1)普通随机森林
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
rfr_y_predict = rfr.predict(x_test)
# r2_score(y_test,rfr_y_predict)
print ('普通随机森林预测准确率为：',r2_score(y_test,rfr_y_predict))
# print (df_test[:10])

# (2)极端随机森林
efr = ExtraTreesRegressor()
efr.fit(x_train,y_train)
efr_y_predict = efr.predict(x_test)
# r2_score(y_test,rfr_y_predict)
print ('极端随机森林预测准确率为：',r2_score(y_test,efr_y_predict))

# （3）人工神经网络
mlp = MLPRegressor()
mlp.fit(x_train,y_train)
mlp_y_predict = mlp.predict(x_test)
print ('人工神经网络预测准确率为：',r2_score(y_test,mlp_y_predict))


# columns_trans= ['season','weather','weekday','hour']
# df = pd.get_dummies(df,columns=columns_trans)
# print(df.info())






# a = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
# b = 'hello'
# D = map(set,[1,2,3])
# print(D)
#
# print('hello')
#
# df = pd.DataFrame(np.arange(12).reshape((3,4)),index=['a','b','c'],columns=['one','two','three','four'])
# df1 = df.drop('one',axis=1)
# df2 = df.sort_values(by='one',ascending = False)
# df3 = df.sort_index(ascending=False)
# print (df)
# print (df1)
# print (df2)
# print (df3)
# df4 = pd.DataFrame(np.arange(120000).reshape(30000,4),index= np.arange(30000),columns=['one','two','three','four'])
# df4.to_csv('C:/Users/guoqingjin/Desktop/test.csv')
