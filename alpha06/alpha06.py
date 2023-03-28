# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:15:47 2023

@author: 123
alpha06: (-1 * correlation(open, volume, 10))
首先计算近10个交易日开盘价和成交量的相关性，相关性越高说明开盘价与成交量
越趋近于“同涨同跌”，但是最终的结果加上负号后，则近10个交易日开盘价与成交
量涨跌方向完全相反因子值反而越高，逻辑就是做多‘价量背离’的股票。
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

def correlation(x,y,window = 10):
    '''
    滑动窗口求x与y的相关系数
    注意，x,y的格式并非双重索引（参照02因子）
    Parameters
    ----------
    x : df(或个中一列)
        过去两日对数交易量差
    y : df(或个中一列)
        今日涨跌幅
    window : int
        滑动窗口期，默认是10天
    Returns
    -------
    '''
    return x.rolling(window).corr(y)

def alpha06(open_df,volume_df):
    '''
    #为后续统一格式，这里的open和volume都是双重索引，所以函数内部要转化
    '''
    x1 = open_df.unstack()
    x2 = volume_df.unstack()
    df = pd.DataFrame(index=x2.index,columns = x2.columns)
    for i in range(0,x1.shape[1]):
        x = -1*correlation(x1.iloc[:,i],x2.iloc[:,i],10)
        df.iloc[:,i] = x#第i列即第i个stock对应的就是这个相关系数序列
    return df.fillna(0).stack()

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
    list(包含IC_mean,IC_std,ic大于0的rate)
    '''
    #这里的alpha和return_df都必须是双重索引的格式
    ic = []
    for i in range(0,len(index)-1):
        ic.append(alpha.loc[index[i]].corr(return_df.loc[index[i+1]]))#index[i]是date
    ic_s = pd.Series(ic)#转化为series便于计算指标
    IC_mean = ic_s.mean()
    IC_std = ic_s.std()
    rate = len(ic_s[ic_s>=0])/len(ic_s)
    stats = [IC_mean,IC_std,rate]
    return ic_s,stats

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
    a = close.pct_change(periods = d).fillna(0)#计算过去d天的滚动收益率
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

def sort(df,column='alpha'):
    return df.sort_values(by = column,ascending = False)#降序排列确保因子值由大到小
#定义sort方法，方便日内对所有股票进行排序(用groupby.apply)默认是升序排序
#现在是降序排列，因此最大因子那层就是stock_list最小的那层
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