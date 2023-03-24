# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:43:45 2023

@author: 123
alpha05:
(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))

数据输入：
open：开盘价
close：收盘价
vwap：平均成交价

1、rank(x)
含义：x元素的排名，取值范围为[0,1]。
例如：
输入五只股票的收盘价[2,7,8,5,9]，返回的是各股票收盘价排名的分位值[0.2, 0.6, 0.8, 0.4, 1]

2、abs (x)
含义：x元素求绝对值

3、sum(x, d)
含义：计算过去d天x的累加值

公式解析：
rank((open - (sum(vwap, 10) / 10)))：首先对过去10个交易日的平均
成交价（vwap）累加后再计算平均值，用开盘价（open）减去过去10日的平均
价，得到的数值的正负代表当日开盘价是否突破10日均线，最后进行排序。
(-1 * abs(rank((close - vwap))))：计算收盘价相对于当日平均成交价
的强弱程度进行排序，这里进行绝对值操作（abs）貌似没有意义，因为rank的
返回值本身就是（0,1]之间的排序百分比，最后加上负号进行反向操作。
(rank((open - (sum(vwap, 10) / 10))) * 
 (-1 * abs(rank((close - vwap)))))：
最后将以上两部分结果相乘，即开盘价越高同时收盘价越低或者开盘价越低
同时收盘价越高的股票对应因子值越大，逻辑上就是做多‘高开低收’和
‘低开高收’的股票，做空‘高开高收’和‘低开低收’的股票。
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

def rank(df):
    '''
    Parameters
    ----------
    df : series
        双重索引
    Returns
    -------
    日内排序
    '''
    return df.groupby(level = 'Date').rank(pct = True)

def Sum(x,d):
    '''
    计算过去d天x的累加值
    Parameters
    ----------
    x : series
        双重索引
    d : int
        过去d天

    Returns
    -------
    series

    '''
    df1 = x.unstack()
    df1 = df1.rolling(d).sum().fillna(0)
    return df1.stack()

def alpha05(open_df,close_df,vwap_df):
    x1 = rank(open_df - (Sum(vwap_df, 10) / 10))
    x2 = -1 * abs(rank((close_df - vwap_df)))
    return x1*x2

def IC_fenxi(alpha,return_df,index):
    '''
    Parameters
    ----------
    alpha : df
        因子表.
    return_df : df
        收益表
    index : 日期索引
    Returns
    -------
    IC值序列以及list(包含IC_mean,IC_std,ic大于0的rate)
    '''
    #这里的alpha和return_df都必须是双重索引的格式
    #alpha.loc[date]则是该date的
    ic = []
    for i in range(0,len(index)-1):
        ic.append(alpha.loc[index[i]].corr(return_df.loc[index[i+1]]))
    ic_s = pd.Series(ic)
    ic_mean = ic_s.mean()
    ic_std = ic_s.std()
    rate = len(ic_s[ic_s>=0])/len(ic_s)   
    stats = [ic_mean,ic_std,rate]
    return ic_s,stats

def sort(df,column = 'alpha'):
    '''
    为了每日按照因子值排序，便于计算分层收益
    '''
    return df.sort_values(by = column,ascending = False)#降序排列

def HuanCangIC(alpha,close_df,d,index):
    '''
    Parameters
    ----------
    alpha : sereis
            双重索引
    close_df:series
            双重索引
    d : int
        换仓周期，最好与换仓分层函数的换仓周期是一样的
    index:日期索引
          用于提取换仓日（这里IC的计算是第i个换仓日的alpha与第i+1个换仓日的return进行corr计算）
    Returns
    -------
    IC seireis and stats list
    '''
    HuanCangRi_index = list(range(0,len(index),d))
    HuanCangRi = [index[x] for x in HuanCangRi_index]
    
    x = alpha.unstack()
    alpha_slice = pd.DataFrame(columns = x.columns)
    for i in HuanCangRi:
        alpha_slice = alpha_slice.append(x.loc[i])#至此获取所有换仓日的alpha
    alpha_slice = alpha_slice.stack()
    
    close = close_df.unstack()
    a = close.pct_change(periods = d).fillna(0)#计算d天收益率
    return_slice = pd.DataFrame(columns = close.columns)

    for i in HuanCangRi:
        return_slice = return_slice.append(a.loc[i])
    return_slice = return_slice.stack()
    #alpha和return都转为stack,以便计算corr时两层for变为一层for循环
    ic = []
    for i in range(len(HuanCangRi)-1):
        ic.append(alpha.loc[HuanCangRi[i]].corr(return_slice.loc[HuanCangRi[i+1]]))
    ic_s = pd.Series(ic)#转化为series便于计算指标
    IC_mean = ic_s.mean()
    IC_std = ic_s.std()
    rate = len(ic_s[ic_s>=0])/len(ic_s)
    stats = [IC_mean,IC_std,rate]
    return ic_s,stats,alpha_slice,return_slice.shift(-1),HuanCangRi 

def HuanCangFenCeng_ShouYi(data_slice,HuanCangRi,n,d):
    '''
    Parameters
    ----------
    data_slice：含有alpha和return两列的双重索引的dataframe
                便于根据alpha sort后直接分层
    HuanCangRi:日期列表
             （储存换仓日）
    n : int
        所分层数
    d : int
        换仓周期
    Returns
    -------
    返回分层收益表和分层累计收益表
    '''
    #数据存储格式：n个list存储全阶段n层收益
    #这里可直接抛开所选股票是什么而获取收益
    #第一步先根据alpha进行排序
    data_slice = data_slice.groupby(level = 'Date').apply(sort)
    #第二步
    for i in range(n):
        globals()['stocklist' + str(i)] = []
    for date in HuanCangRi:
        a = data_slice.loc[date]['return']#a是当天的所有股票的return
        for j in range(n):
            globals()['stocklist' + str(j)].append(sum(a[math.ceil(len(a)*j*1/n):math.ceil(len(a)*(j+1)*1/n)])/math.ceil(len(a)/n))#1个stock投len(a)/n即等权，以防数据过大
        #每层list都append对应那层的sum of return
    HuanCangFenCeng_return = pd.DataFrame(index = HuanCangRi)
    for i in range(n):
        HuanCangFenCeng_return['stocklist_' + str(i)] = globals()['stocklist' + str(i)]
    #至此就完成了分层收益的df
    #由于简单收益率不能像对数收益率一样直接相加，因此需要先加1再连乘至要计算的那一期，最后减1
    Cumsum_return = (HuanCangFenCeng_return+1).cumprod() - 1

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
        plt.plot(Cumsum_return.index,Cumsum_return.iloc[:,i])
    plt.legend(legend_list0)#图例
    plt.xlabel('Time')
    plt.ylabel('分层累计收益')
    plt.xticks(rotation = 30)
    plt.xticks(ticks = Cumsum_return.index)
    
    return HuanCangFenCeng_return,Cumsum_return