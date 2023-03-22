# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:39:01 2023

@author: 123
alpha004
公式:(-1 * Ts_Rank(rank(low),9))
说明：Ts_Rank是时间序列上的排序,只返回最后一天(t-1)的值，
区别于rank在横截面数据上的排序。对所有股票计算low的绝对值排序，
对所有股票计算过去9天中low排序的排序。生成的是每一只股票，
在t-1的时候，low在t-9内的百分位数。相当于如果low在t-1的时候，
low的绝对值大小排名有所下降(变得更加便宜)，则购买。
也就是对于每个stock而言，其过去9天的最低价横截面排名做一次时序排名
"""
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.stats import rankdata

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
    
#在过去d天的时序排名
def rolling_rank(na):
    #因为df的rolling项不允许用rank方法，因此需要该函数用于apply
    #返回的是最后一项即t-1天的排名
    #rankdata为数据分配等级，返回的是值数组，里面是排名分数
    return rankdata(na)[-1]

def ts_rank(df,window):
    '''
    对数据在时间序列上排序而非是在横截面上进行排序
    注意只返回在t-1天的排名值
    Parameters
    ----------
    df : series
        双重索引
    window : int
        时间窗口

    Returns
    -------
    None.

    '''
    a = rank(df).unstack()#变回行日期列stock的格式，便于时序排序
    b = a.rolling(window).apply(rolling_rank)
    return b.fillna(0).stack()#变为双重索引series

def alpha04(low):
    #low是最低价的series
    alpha = -1*ts_rank(low,9)
    return alpha

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
    Cumsum_return = HuanCangFenCeng_return.cumsum()#寻找每列的总和，axis = None
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
