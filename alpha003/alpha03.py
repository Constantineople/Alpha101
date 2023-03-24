# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:48:15 2023

@author: 123
alpha03因子
公式：(-1 * correlation(rank(open), rank(volume), 10))
因子函数说明：

1、correlation(x,y,d)
含义：x,y两个随机变量过去d天的相关系数，取值范围[-1,1]。

2、rank(x)
含义：x元素的分位排名，取值范围为[0,1]。
例如：
输入五只股票的收盘价[2,7,8,5,10]，返回的是各股票收盘价排名的分位值[0.2, 0.7, 0.8, 0.5, 1]

需要数据:open volume
先对开盘价和成交量进行排序，然后计算排序结果过去10天的相关系数，最后再乘-1
因此该因子评估的是短期内开盘价和成交量的相关性高低，然后加上负号进行颠倒处理
即希望构建开盘价与成交量相关性低的股票组合，即做多开盘价与成交量产生背离的股票

"""
import pandas as pd
import math
from matplotlib import pyplot as plt

def correlation(x,y,window):
    '''
    滑动窗口求x与y的相关系数
    Parameters
    ----------
    x : df或各中一列
    y : df或各中一列，形式与x相同
    window : int
        滑动窗口

    Returns
    -------
    float

    '''
    return x.rolling(window).corr(y)

def rank(df):
    '''
    Parameters
    ----------
    x : df
        应传入双重索引的df单列
    Returns
    -------
    dataframe
    排序并返回百分位数
    '''
    #按date groupby按天分组，即在一日内的N只股票为一组，后面进行rank
    return df.groupby(level = 'Date').rank(pct = True)

def alpha03(open_df,volume_df):
    '''
    Parameters
    ----------
    open_df : df
        DESCRIPTION.
    volume_df : df
        DESCRIPTION.

    Returns
    -------
    dataframe(因子表)
    '''
    x1 = rank(open_df)
    x2 = rank(volume_df)
    x3 = -1*correlation(x1,x2,10)
    return x3

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
    
    x = alpha.unstack()#
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