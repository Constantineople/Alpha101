# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:39:03 2023

@author: 123
"""
import os 
from alpha04 import *

return_df = pd.read_csv(os.path.join(os.path.abspath('..'),'return_df.csv'),index_col = 0)
close_df = pd.read_csv(os.path.join(os.path.abspath('..'),'close_df.csv'),index_col = 0)
low_df = pd.read_csv(os.path.join(os.path.abspath('..'),'low_df.csv'),index_col = 0)

#给index取名 df.index.name = '名字'
#给列名取名 df.columns.name = '名字'
low_df.index.name = 'Date'
return_df.index.name = 'Date'
close_df.index.name = 'Date'

index = return_df.index#index为日期索引

low_df.columns.name = 'stock_name' 
return_df.columns.name = 'stock_name'
close_df.columns.name = 'stock_name'

#注意stack可直接变为双重索引，将列透视到行产生一个series
low_df = low_df.stack()
return_df = return_df.stack()
close_df = close_df.stack()

data = pd.concat([low_df,close_df,return_df],axis = 1,ignore_index = False)
#axis = 1代表着列拼接，concat 将这几列拼接为同一个df
data.columns = ['low','close','return']

#构建因子表并输出为csv，并将alpha和return一起作为df的slice
alpha = alpha04(low_df)
alpha.unstack().to_csv('alpha.csv')
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