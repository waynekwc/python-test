from finlab.data import Data
import pandas as pd
from finlab import ml
import talib
import time
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

tStart = time.time()

data = Data()

data_start_date = '2020-01-01'
excel_output_date = datetime.now().strftime('%Y-%m-%d')

#add comment2
#add comment
#excel_output_date = '2021-07-27'
#excel_output_date = '2021-01-15'# need to be larger than data_start_date

# print(close_price[(close_price.index<=close_price.index[-2]) & (close_price.index>=close_price.index[-5])])
#add comment3

equity_con = data.get('股本合計')/10/10000*100
equity_con = equity_con[(equity_con.index>=data_start_date) & (equity_con.index<=excel_output_date)].copy()

close_price = data.get("收盤價")
close_price = close_price[(close_price.index>=data_start_date) & (close_price.index<=excel_output_date)].copy()
close_price = close_price.fillna(method='backfill')

open_price = data.get("開盤價")
open_price = open_price[(open_price.index>=data_start_date) & (open_price.index<=excel_output_date)].copy()
open_price = open_price.fillna(method='backfill')

price_jumpratio =open_price/(close_price.shift(1).fillna(method='backfill'))

volume = data.get("成交股數")/1000
volume = volume[(volume.index>=data_start_date) & (volume.index<=excel_output_date)].copy()
volume = volume.fillna(method='backfill')

volume_5_mean = volume.rolling(5).mean()
volume_5_mean_diff1 = volume_5_mean.diff(1)
volume_5_mean_shift1 = volume_5_mean.shift(1)
volume_5_mean_shift1_diff1 = volume_5_mean_shift1.diff(1)

volume_20_mean = volume.rolling(20).mean()
volume_20_mean_diff1 = volume_20_mean.diff(1)
volume_20_mean_shift1 = volume_20_mean.shift(1)
volume_20_mean_shift1_diff1 = volume_20_mean_shift1.diff(1)

close_price_5_mean = close_price.rolling(5).mean()
close_price_200_mean = close_price.rolling(200).mean()
close_price_200_max_09 = close_price.rolling(200).max()*0.9

intrust = (data.get("投信買賣超股數").fillna(0)/1000).round(0).astype(int)
intrust = intrust[(intrust.index>=data_start_date) & (intrust.index<=excel_output_date)].copy()
intrust_y = intrust.shift(1)

selfem = (data.get("自營商買賣超股數(自行買賣)").fillna(0)/1000).round(0).astype(int)
selfem = selfem[(selfem.index>=data_start_date) & (selfem.index<=excel_output_date)].copy()
selfem_y = selfem.shift(1)

foreign = (data.get("外陸資買賣超股數(不含外資自營商)").fillna(0)/1000).round(0).astype(int)
foreign = foreign[(foreign.index>=data_start_date) & (foreign.index<=excel_output_date)].copy()
foreign_y = foreign.shift(1)

features = {
    'close_price': close_price,
    'open_price': open_price,
    'price_jumpratio': price_jumpratio,
    'volume': volume,
    'close_price_5_mean': close_price_5_mean,
    'close_price_200_mean': close_price_200_mean,
    'close_price_200_max_09': close_price_200_max_09,
    'intrust': intrust,
    'intrust_y': intrust_y,
    'selfem': selfem,
    'selfem_y': selfem_y,
    'foreign': foreign,
    'foreign_y': foreign_y,
    'volume_5_mean': volume_5_mean,
    'volume_5_mean_diff1': volume_5_mean_diff1,
    'volume_5_mean_shift1_diff1': volume_5_mean_shift1_diff1,
    'volume_20_mean': volume_20_mean,
    'volume_20_mean_diff1': volume_20_mean_diff1,
    'volume_20_mean_shift1_diff1': volume_20_mean_shift1_diff1,
    'equity_con': equity_con,
}

trading_frequency = close_price.index

for name, f in features.items():
    features[name] = f.reindex(trading_frequency, method='ffill')
    
for name, f in features.items():
    features[name] = f.unstack()
    
dataset = pd.DataFrame(features)
dataset = dataset.reset_index().set_index("date")

datasettest = dataset[dataset['stock_id']>='1101'].copy()

tempdf = datasettest.copy()

# print(close_price[(close_price.index<=close_price.index[-2]) & (close_price.index>=close_price.index[-5])])

#con0 = (tempdf.index <= close_price.index[-2]) & (tempdf.index >= close_price.index[-40])
con00 = (tempdf.index == close_price.index[-1])
con0 = (tempdf.index == close_price.index[-2])
con1 = (tempdf['close_price'] > tempdf['close_price_200_max_09'])
con2 = (tempdf['close_price'] > tempdf['close_price_5_mean'])
con3 = (tempdf['close_price'] > tempdf['close_price_200_mean'])
con4 = (tempdf['volume_5_mean_diff1'] > 0)
con5 = (tempdf['volume_5_mean_shift1_diff1'] > 0)
con6 = (tempdf['volume_20_mean_diff1'] > 0)
con7 = (tempdf['volume_20_mean_shift1_diff1'] > 0)
con8 = (tempdf['volume_5_mean']/tempdf['equity_con'] > 3)
con9 = ~np.isnan(tempdf['close_price'])
con11 = ((tempdf['intrust']>0) | (tempdf['intrust_y']>0)) & ((tempdf['intrust']>=0) & (tempdf['intrust_y']>=0))
con12 = ~((tempdf['selfem']<0) & (tempdf['selfem_y']<0))
con13 = ~((tempdf['foreign']<0) & (tempdf['foreign_y']<0))

price_jumpratio_con = (tempdf['price_jumpratio'] <= 1.05)
#con10 = (tempdf.index != '2021-01-08')
tomorrow_buy_list_con = (con00 & con2 & con6 & con7 & con8 & con9 & con11 & con12 & con13)
tomorrow_df = tempdf.copy()
tomorrow_df.loc[:,'buycond'] = tomorrow_buy_list_con
tomorrow_df_buy = (tomorrow_df.loc[(tomorrow_df['buycond']==1),:])

filename1 = close_price.index[-1].strftime('%Y-%m-%d')+' buy_list_tomorrow.csv'
tomorrow_df_buy.to_csv('/Users/waynechen/Desktop/stock_list_autoupdate/day_trade_strategy5/'+filename1)


# buycon = (con0 & con1 & con2 & con3 & con4 & con5 & con6 & con7 & con9 & con10).astype(int) #(con0 & con1 & con6 & con5).astype(int)
# buycon = (con0 & con1 & con2 & con3 & con6 & con7 & con8 & con9).astype(int)
#buycon = (con0 & con1 & con2 & con3 & con6 & con7 & con8 & con9).astype(int)
buycon = (con0 & con2 & con6 & con7 & con8 & con9 & con11 & con12 & con13)

tempdf.loc[:,'buycond'] = buycon

# print('stock number = ', len(tempdf[tempdf['buycond']==1]))
# print(tempdf[tempdf['buycond']==1])
print('stock number = ', len(tempdf[tempdf['buycond']]))
print(tempdf[tempdf['buycond']])

tempdf.loc[:,'buy'] = ((tempdf['buycond'].shift(1))&price_jumpratio_con).astype(int).fillna(0)

tempdf.loc[:,'buyprice'] = (tempdf['buy'] == 1).astype(int) * tempdf['open_price']
tempdf.loc[:,'sellprice'] = (tempdf['buy'] == 1).astype(int) * tempdf['close_price']

tempdf.loc[:,'buyprice':'sellprice'] = tempdf.loc[:,'buyprice':'sellprice'].fillna(0)

print( 'sellprice len = ', len(tempdf[tempdf['sellprice']!=0]))
print( 'buyprice len = ',len(tempdf[tempdf['buyprice']!=0]))

# (tempdf[tempdf['sellprice']!=0]).to_csv('sellprice.csv')
# (tempdf[tempdf['buyprice']!=0]).to_csv('buyprice.csv')

tempdf.loc[tempdf['sellprice'] != 0,'profit'] = np.round((((tempdf[tempdf['sellprice']!=0])['sellprice'].values/(tempdf[tempdf['buyprice']!=0])['buyprice'].values)-(0.0025)),5)-1
tempdf.loc[tempdf['sellprice'] == -1,'profit'] = 0

tempdf.loc[tempdf['sellprice'] != 0,'buydate'] = ((tempdf[tempdf['buyprice']!=0]).index).astype(str)
tempdf.loc[tempdf['sellprice'] != 0,'buyprice'] = (tempdf[tempdf['buyprice']!=0])['buyprice'].values
tempdf.loc[:,'tradingcount'] = abs(tempdf.loc[:,'buy'])

# tempdf.to_csv('checkprofit.csv', encoding='utf_8_sig')
# checkvaliddf = (tempdf.loc[((tempdf['buycond']==1)|(tempdf['buy']==1)),:])
# checkvaliddf.to_csv('checkvaliddf.csv', encoding='utf_8_sig')

# summarydf = tempdf.groupby('stock_id').agg({'tradingcount':'sum','profit':'mean'})

# summarydf.to_csv('summarydf.csv')

trading_record_df = (tempdf.loc[(tempdf['buy']==1),:])

# '/home/username/output.txt'

filename2 = close_price.index[-1].strftime('%Y-%m-%d')+'_trading_record_today.csv'

trading_record_df.to_csv('/Users/waynechen/Desktop/stock_list_autoupdate/day_trade_strategy5/'+filename2)

trading_record_df_temp = trading_record_df.copy()

trading_record_df_temp['colors'] = ['green' if x < 0 else 'red' for x in trading_record_df_temp['profit']]
trading_record_df_temp['profit'] = np.round(trading_record_df_temp['profit']*100,2)
trading_record_df_temp.sort_values('profit', inplace=True)
trading_record_df_temp.reset_index(inplace=True)

plt.figure(figsize=(14,10), dpi= 80)
plt.hlines(y=trading_record_df_temp.index, xmin=0, xmax=trading_record_df_temp.profit, color=trading_record_df_temp.colors, alpha=0.4, linewidth=10)

for x, y, tex in zip(trading_record_df_temp.profit, trading_record_df_temp.index, trading_record_df_temp.profit):
    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'green' if x < 0 else 'red', 'size':14})

# Decorations
plt.gca().set(ylabel='stockid', xlabel='day trading profit('+'%'+'$)$')
plt.yticks(trading_record_df_temp.index, trading_record_df_temp.stock_id, fontsize=15)
plt.title(close_price.index[-1].strftime('%Y-%m-%d')+' stockid versus day trading profit', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.savefig('/Users/waynechen/Desktop/stock_list_autoupdate/day_trade_strategy5/'+close_price.index[-1].strftime('%Y-%m-%d')+'_trading_record_today.jpg')

print('average profit = ', tempdf['profit'].mean())

tEnd = time.time()
print( 'function cost %f sec' %(tEnd - tStart))  
