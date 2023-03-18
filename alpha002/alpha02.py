# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:41:54 2023

@author: 123
alpha101因子 02因子

02因子公式:
(-1 * corr(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
函数说明
1.corr(x,y,d)
含义是x,y过去d天的相关系数，范围从-1到1

2.rank(x)x元素的排名，范围0到1
例如：
输入五只股票的收盘价[2,7,8,5,9]，返回的是各股票收盘价
排名的分位值[0.2, 0.6, 0.8, 0.4, 1]

3.delta(x,d)
含义：当天x的值减去过去第d天x的值
例如：计算股票i最新收盘价减去20天前的收盘价:
delta(xi, 20)，其中xi是股票i的收盘价时间序列。

所需公用变量：volume成交量，close收盘价，open开盘价

公式解析：
rank(delta(log(volume), 2))：成交量取对数可以降低成
交量的数量级，减小不同股票成交量数量级的差异；对数化后的
成交量依次与两天前进行比较可以获得股票成交量的趋势：大于
零代表成交量近两日上涨反之下降；对股票成交量变化趋势进行
排序。也是对所有股票排序,ps:对数列取对数用np.log

rank(((close - open) / open))：对当日股票涨跌幅进行
排序。注意是对当日所有股票排序

correlation（rank(delta(log(volume), 2)), 
rank(((close - open) / open)), 6）：
对以上两个指标的排序结果计算近6日的相关系数，所以该因子
评估的是短期成交量趋势与涨跌幅的相关性高低，最后加上负号
则将相关性进行了颠倒处理，为的是构建投资组合时买入成交量
与涨跌幅相关性低的股票，内在逻辑就是买入价量背离的股票。
"""

'''
与01因子不同，这里由于数据种类不止是close1种了，因此采
用多重索引更好，我们用日期作为第一层索引，用stock_name
作为第二层索引，因此rank等函数要发生变化了，对于rank来
说，我们用groupby(level = 日期)，注意日期列index应有
name为日期，这样就可以实现日内所有股票进行排名
'''
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import math
import os



def rank(df):
    '''
    Parameters
    ----------
    df : dataframe
         应传入已经对日期groupby好的表格（单列）
    Returns
    -------
    dataframe
    排序并返回百分位数
    '''
    #pct = True将排序值设为0-1之间的数
    return df.groupby(level = 'Date').rank(pct = True)

def correlation(x,y,window = 6):
    '''
    滑动窗口求x与y的相关系数
    Parameters
    ----------
    x : df(或个中一列)
        过去两日对数交易量差
    y : df(或个中一列)
        今日涨跌幅
    window : int
        滑动窗口期，默认是6天
    Returns
    -------
    '''
    return x.rolling(window).corr(y)

def delta(x,d):
    '''
    Parameters
    ----------
    x : dataframe（双重索引）
        对数交易量
    d : shift窗口
        过去第几日

    Returns
    -------
    返回一个d阶差分
    '''
    #若x是序列用此return x-x.shift(d)
    #x是df可以直接用diff函数
    return x.groupby(level = 'stock_name').diff(d)

close_df = pd.read_csv(os.path.join(os.path.abspath('..'),'close_df.csv'),index_col = 0)
open_df = pd.read_csv(os.path.join(os.path.abspath('..'),'open_df.csv'),index_col = 0)
volume_df = pd.read_csv(os.path.join(os.path.abspath('..'),'volume_df.csv'),index_col = 0)
return_df = pd.read_csv(os.path.join(os.path.abspath('..'),'return_df.csv'),index_col = 0)
#给index取名 df.index.name = '名字'
#给列名取名 df.columns.name = '名字'

close_df.index.name = 'Date'
open_df.index.name = 'Date'
volume_df.index.name = 'Date'
return_df.index.name = 'Date'

index = return_df.index

close_df.columns.name = 'stock_name'
open_df.columns.name = 'stock_name'
volume_df.columns.name = 'stock_name'
return_df.columns.name = 'stock_name'

#注意stack可直接变为双重索引，将列透视到行产生一个series
close_df = close_df.stack()    
open_df = open_df.stack()
volume_df = volume_df.stack()
return_df = return_df.stack()
#以上三个都变成了series，将它们合并为一个dataframe
data = pd.concat([open_df,close_df,volume_df,return_df],axis = 1,ignore_index = False)
#axis = 代表着列拼接
data.columns = ['open','close','volume','return']
#columns重新命名

#因子公式
#(-1 * corr(rank(delta(log(volume), 2)), 
#           rank(((close - open) / open)), 6))
def alpha02(open_df,close_df,volume_df):
    x1 = rank(delta(np.log(volume_df),2)).unstack()
    x2 = rank((close_df-open_df)/open_df).unstack()
    #由于内存限制等各种原因，计算因子还是得通过单索引表循环方式
    df = pd.DataFrame(index=x2.index,columns = x2.columns)
    for i in range(0,x1.shape[1]):
        x = -1*correlation(x1.iloc[:,i],x2.iloc[:,i],6)
        df.iloc[:,i] = x
    return df.fillna(value = 0).stack()

alpha = alpha02(open_df,close_df,volume_df)
alpha.to_csv('alpha.csv')
data['alpha'] = alpha
data_slice = data[['alpha','return']]

def IC_fenxi(alpha,return_df,index):
    #这里的alpha和return_df都必须是双重索引的格式
    ic = []
    for i in range(0,len(index)-1):
        ic.append(alpha.loc[index[i]].corr(return_df.loc[index[i+1]]))
    ic_s = pd.Series(ic)#转化为series便于计算指标
    IC_mean = ic_s.mean()
    IC_std = ic_s.std()
    rate = len(ic_s[ic_s>=0])/len(ic_s)
    stats = [IC_mean,IC_std,rate]
    return ic_s,stats

IC_s,IC = IC_fenxi(alpha,return_df,index)
#对IC画一个直方图
plt.figure(dpi = 300,figsize = (24,8))#dpi调分辨率,figsize调画布长宽
plt.hist(IC_s,bins=400)
plt.show()

def sort(df,column='alpha'):
    return df.sort_values(by = column,ascending = False)#降序排列确保因子值由大到小
#定义sort方法，方便日内对所有股票进行排序(用groupby.apply)默认是升序排序
#现在是降序排列，因此最大因子那层就是stock_list最小的那层
def HuanCangFenCeng_ShouYi(data_slice,index,n,d):
    '''
    Parameters
    ----------
    data_slice：含有alpha和return两列的双重索引的dataframe
                便于根据alpha sort后直接分层
    
    index : list
            日期list
    
    n : int
        所分层数
    d : int
        换仓周期
    Returns
    -------
    只剩下换仓日中分层的df
    思路是首先要做一个每日分层收益的df
    '''
    #首先做每日分层收益df
    #数据存储格式：n个list存储全阶段n层收益
    #这里可直接抛开所选股票是什么而获取收益
    #第一步先根据alpha进行排序
    data_slice = data_slice.groupby(level = 'Date').apply(sort)
    #第二步
    for i in range(n):
        globals()['stocklist' + str(i)] = []
    for date in index:
        a = data_slice.loc[date]#a是当天的所有股票数据
        for j in range(n):
            globals()['stocklist' + str(j)].append(sum(a[math.ceil(len(a)*j*0.1):math.ceil(len(a)*(j+1)*0.1)]['return']))
        #每层list都append对应那层的sum of return
    Daily_FenCeng_df = pd.DataFrame(index = index)
    for i in range(n):
        Daily_FenCeng_df['stocklist_' + str(i)] = globals()['stocklist' + str(i)]
    #至此就完成了每日分层收益的df
    
    #接下来要根据换仓时间在df中进行选取
    HuanCangRi_index = list(range(0,(len(index)),d))#存放要取用的日期
    HuanCangRi = [index[x] for x in HuanCangRi_index]
    HuanCangFenCeng_return = pd.DataFrame()
    for i in range(n):
        HuanCangFenCeng_return['stocklist_' + str(i)] = []
    for j in HuanCangRi:
        HuanCangFenCeng_return = HuanCangFenCeng_return.append(Daily_FenCeng_df.loc[j])
        #append直接添加一整行，需注意，要令前面=才可，否则append不会生效
    #HuanCangFenCeng_return.index = HuanCangRi
    #画图
    plt.figure(dpi = 300,figsize=(24,8))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    legend_list0 = []
    for i in range(n):
        b = 'stocklist_' + str(i)
        legend_list0.append(b) 
    for i in range(n):
        plt.plot(HuanCangFenCeng_return.index,HuanCangFenCeng_return.iloc[:,i])
    plt.legend(legend_list0)#图例
    plt.xlabel('Time')
    plt.ylabel('分层收益')
    plt.xticks(rotation = 30)
    plt.xticks(labels=HuanCangRi)
    
    return HuanCangFenCeng_return
HuanCangFenCeng_return = HuanCangFenCeng_ShouYi(data_slice,index,10,5)