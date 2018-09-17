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
from sklearn.feature_extraction import DictVectorizer
import matplotlib as plt
 
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

# 导入数据并转化为DataFrame
data = pd.read_csv('C:/Users/leroy/IdeaProjects/bike_kaggle/train.csv',header= 0,error_bad_lines= False)
df = pd.DataFrame(data)
df_isnull(df)

df = time_clean(df)
# print (df[:10])

sep_list= ['season','weather','weekday','hour']
df = sep_dummies(df,sep_list)

conti_list = ['temp','atemp','humidity','windspeed']
conti_standard(df,conti_list)
print (df[:10])


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
