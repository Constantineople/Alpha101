# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:51:01 2023

@author: 123
阿尔法101因子编写
"""

'''
alpha01因子
公式：
rank(Ts_ArgMax(SignedPower(((returns < 0) ? 
stddev(returns, 20) : close), 2), 5)) - 0.5

1.rank(x)股票排名，输入值向量为股票向量，若输入值含NAN，则NAN
不参与排名输出为股票对应排名的Boolean值（排名所占总位数的百分比）

2.Ts_ArgMax(x,d)，x为向量，找出前d天的向量x值最大的值
例如：
d=5 , x=[5,8,9,3,2]，对应索引号为L=[5,4,3,2,1]，
其中，索引号为5代表过去第5天，索引号为1代表过去第1天。
因为x中最大的元素为9，则索引号为3，即过去第3天。

3.SignedPower(x,t)=Sign(x).*(Abs(x).^t)
保持向量x 的正负特性，将x进行t次幂处理使其差异放大。
其中，Sign(x)为符号函数，表示：如果x>0，就返回1，如果x<0，
则返回-1，如果x=0，则返回0。
Abs(x)为绝对值函数，进行非负数处理。

4.stddev(x,n)
含义：求前 n 个 x 值的标准差。

01的输入是returns和close，前者是收益率向量，输入n+1行close
输出n行returns，后者是收盘价

结构拆解：
1.x1 = (returns < 0 ? stddev(returns, 20) : close)
判断当returns<0成立则返回stddev(returns, 20)，即判断每日
收益率若＜0就返回过去20天收益率的标准差，否则返回close。

2.x2 = SignedPower(x1,2)
对x1进行保留正负号的平方处理，进行差异放大，原因是x1对应的值为
收盘价和前20天的回报率的标准差两种。将其差异放大之后变成x2，
此时，收盘价的平方普遍大于前20天的回报率的方差。
(std不可能为负，而<0的return都被替换了，因此直接平方即可)

3.x3 = Ts_ArgMax(x2,5)
找出过去5个交易日x2的最大值，返回其对应索引，即找出过去5天最大
的收盘价或者最大的前20天标准差在哪一天。x2可能全为收盘价，全为
方差，或两者都有。因为进行了差异放大，因此只要有returns>0基本
可以确定过去5天最大值是close。
下行波动率最高的一天离计算时间越近，越可以投资。
收盘价最高离计算时间越近，越可以投资。

4.x4 = rank(x3-0.5)
先排序再进行-0.5中性化变换
x3的值为各股票根据前5天最大收盘价或最大的前20天的回报率的标准
差的索引作为对应股票的权重值，那么对其进行排序以及-0.5中性化，
这就是01因子。取正数的股票为买入池，负数的为卖出池，
构建多空组合。即对所有股票的x3值进行排序
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
    #用level = 'Date'则是按照date索引进行排序，即在当天对所有股票进行排序
    return df.groupby(level = 'Date').rank(pct = True)

def stddev(df,window):
    #滑动窗口求标准差
    df1 = df.groupby(level = 'stock_name').rolling(window).std()
    #此时index是三重，为stock_name,date,stock_name，需要删掉第0个
    df1.index = df1.index.droplevel(0)
    #注意若不fillna最开始的window-1天的数据将会被丢掉
    return df1.fillna(0).unstack().stack()#因为drop掉第0个index后索引是一日对一个股票，要恢复到原本的双重索引格式要先unstack变为行列索引再stack变为双重索引

def Ts_ArgMax(df,window):
    '''
    滑动窗口中的数据最大值位置
    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    window : int
        窗口
    Returns
    -------
    None.
    '''
    #返回最大值索引，因为从0开始所以要加1
    df1 = df.unstack().rolling(window).apply(np.argmax) + 1#双重索引的计算时间太长
    return df1.fillna(0).stack()#最开始的window-1个数将会被丢掉

close_df = pd.read_csv(os.path.join(os.path.abspath('..'),'close_df.csv'),index_col = 0)
return_df = pd.read_csv(os.path.join(os.path.abspath('..'),'return_df.csv'),index_col = 0)
#给index取名 df.index.name = '名字'
#给列名取名 df.columns.name = '名字'

close_df.index.name = 'Date'
return_df.index.name = 'Date'

index = return_df.index#日期索引（在return变为双重索引前先行储存）

close_df.columns.name = 'stock_name'
return_df.columns.name = 'stock_name'

#注意stack可直接变为双重索引，将列透视到行产生一个series
close_df = close_df.stack()    
return_df = return_df.stack()
#以上三个都变成了series，将它们合并为一个dataframe
data = pd.concat([close_df,return_df],axis = 1,ignore_index = False)
#axis = 1代表着列拼接
data.columns = ['close','return']
#columns重新命名


def alpha01(return_df,close_df):
    std_df = stddev(return_df,20)
    x1 = return_df
    for i in range(len(x1)):
        if return_df[i]<0:
            x1[i] = std_df[i]
        elif return_df[i]>0:
            x1[i] = close_df[i]
    x2 = x1 ** 2
    x3 = Ts_ArgMax(x2, 5).fillna(0)
    x4 = rank(x3)-0.5
    alpha = x4.fillna(0)
    return alpha

alpha = alpha01(return_df,close_df)
alpha.unstack().to_csv('alpha.csv')
data['alpha'] = alpha

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

IC_s,IC = IC_fenxi(alpha,return_df,index)
#对IC画一个直方图
plt.figure(dpi = 300,figsize = (24,8))#dpi调分辨率,figsize调画布长宽
plt.hist(IC_s,bins=400)
plt.show()

def sort(df,column='alpha'):
    return df.sort_values(by = column,ascending = False)#降序排列确保因子值由大到小
#定义sort方法，方便日内对所有股票进行排序(用groupby.apply)默认是升序排序，用于换仓分层收益的计算
#因此最大因子那层就是stock_list最大的那层

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

HuanCang_IC_s,HuanCang_IC,alpha_slice,return_slice,HuanCangRi = HuanCangIC(alpha,close_df,5,index)

data_slice =pd.DataFrame()
data_slice['alpha'] = alpha_slice
data_slice['return'] = return_slice
data_slice.dropna(inplace = True)
data_slice.index.names = ['Date','stock_name']


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
HuanCangFenCeng_return,Cumsum_return = HuanCangFenCeng_ShouYi(data_slice,HuanCangRi,10,5)               


    