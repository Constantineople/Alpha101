# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:41:39 2023

@author: 123
"""

import os 
from alpha03 import *
'''
这一部分是为了将csv文件转化为双重索引
'''
open_df = pd.read_csv(os.path.join(os.path.abspath('..'),'open_df.csv'),index_col = 0)
volume_df = pd.read_csv(os.path.join(os.path.abspath('..'),'volume_df.csv'),index_col = 0)
return_df = pd.read_csv(os.path.join(os.path.abspath('..'),'return_df.csv'),index_col = 0)
close_df = pd.read_csv(os.path.join(os.path.abspath('..'),'close_df.csv'),index_col = 0)

#给index取名 df.index.name = '名字'
#给列名取名 df.columns.name = '名字'
open_df.index.name = 'Date'
volume_df.index.name = 'Date'
return_df.index.name = 'Date'
close_df.index.name = 'Date'

index = return_df.index#index为日期索引

open_df.columns.name = 'stock_name' 
volume_df.columns.name = 'stock_name'
return_df.columns.name = 'stock_name'
close_df.columns.name = 'stock_name'

#注意stack可直接变为双重索引，将列透视到行产生一个series
open_df = open_df.stack()
volume_df = volume_df.stack()
return_df = return_df.stack()
close_df = close_df.stack()

data = pd.concat([open_df,volume_df,return_df],axis = 1,ignore_index = False)
#axis = 1代表着列拼接，concat 将这几列拼接为同一个df
data.columns = ['open','volume','return']

#构建因子表并输出为csv，并将alpha和return一起作为df的slice
alpha = alpha03(open_df,volume_df)
alpha.to_csv('alpha.csv')
data['alpha'] = alpha


#获取IC的序列以及相关指标
IC_s,IC = IC_fenxi(alpha,return_df,index)
#对IC画一个直方图
plt.figure(dpi = 300,figsize = (24,8))#dpi调分辨率,figsize调画布长宽
plt.hist(IC_s,bins=400)
plt.show()

HuanCang_IC_s,HuanCang_IC,alpha_slice,return_slice,HuanCangRi = HuanCangIC(alpha,close_df,1,index)

data_slice =pd.DataFrame()
data_slice['alpha'] = alpha_slice
data_slice['return'] = return_slice
data_slice.dropna(inplace = True)
data_slice.index.names = ['Date','stock_name']

HuanCangFenCeng_return,Cumsum_return = HuanCangFenCeng_ShouYi(data_slice,HuanCangRi,10,1)