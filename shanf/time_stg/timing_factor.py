import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import seaborn as sns
from numpy.linalg import inv
import talib
import empyrical
from scipy.stats import chi2
from utils import *
from WindPy import *
sns.set(style='white')
plt.rcParams['font.family'] = 'HeiTi TC'
plt.rcParams['font.size'] = 20
warnings.filterwarnings('ignore')

def get_tech_indicator_stats(df, col, start=2016,end=2024):
    df = df[f'{start}-01-01':f'{end}-01-01']
    sig, trade = df[f'sig_{col}'], df[f'trade_{col}']
    ret = df['return']
    stg_ret = sig *np.array(ret)
    long_ret = stg_ret[stg_ret>0]
    short_ret = stg_ret[stg_ret<0]
    stg_ret =(1 + np.array(stg_ret)) - 1 ## transaction cost adjusted returns 

    def mycorr(x,y):
        return np.ma.corrcoef(np.ma.masked_invalid(x), np.ma.masked_invalid(y))[0][1]
    
    ## METRICS 
    corr = mycorr(sig,ret) 
    ann_ret = empyrical.annual_return(stg_ret, period='daily')
    sharpe = empyrical.sharpe_ratio(stg_ret, period='daily')  
    calmar_ratio = empyrical.calmar_ratio(stg_ret) 
    max_dd = empyrical.max_drawdown(stg_ret)  
    win_ratio_day = np.where(stg_ret>0,1,0).sum() / (np.where(stg_ret>0,1,0).sum() + np.where(stg_ret<0,1,0).sum()) 
    avg_holding_day = np.count_nonzero(sig == 1)/np.count_nonzero(trade > 0) if np.count_nonzero(trade>0) > 0 else 0
    avg_trade_count = (125*np.count_nonzero(~np.isnan(trade)&(trade!=0)))/len(sig)

    ## Deal with nan
    corr = corr if not np.isnan(corr) and isinstance(np.nan, float) and type(corr) != np.ma.core.MaskedConstant  else -1.0
    ann_ret = ann_ret if not np.isnan(ann_ret) else -np.inf
    sharpe = sharpe if not np.isnan(sharpe) else -np.inf
    calmar_ratio = calmar_ratio  if not np.isnan(calmar_ratio) else -1
    max_dd = max_dd  if not np.isnan(max_dd) and max_dd < 0 else -1.0
    win_ratio_day = win_ratio_day if not np.isnan(ann_ret) else 0
    # win_ret_ratio = win_ret_ratio if not np.isnan(win_ret_ratio) else 0
    avg_holding_day = avg_holding_day

    # 次胜率 
    temp_df = df.copy()
    diff = temp_df[f'sig_{col}'] != temp_df[f'sig_{col}'].shift(1)
    temp_df[f'sig_{col}'+'_diff'] = diff.cumsum()
    cond = temp_df[f'sig_{col}'+'_diff'] != 1
    # 每次开仓的收益率情况
    temp_df = temp_df[cond].groupby(f'sig_{col}'+'_diff')['return'].sum()
    # 盈利次数
    win_count = np.sum(np.where(temp_df > 0, 1, 0))
    # 亏损次数
    lose_count = np.sum(np.where(temp_df < 0, 1, 0))
    trade_count = win_count+lose_count
    # 多头天数
    long_count = np.sum(np.where(df[f'sig_{col}']>0, 1, 0))
    short_count = np.sum(np.where(df[f'sig_{col}']<0, 1, 0))
    
    win_ratio_by_trade = win_count/(win_count+lose_count)
    mean_win = np.sum(np.where(temp_df > 0, temp_df, 0))/win_count
    mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0))/lose_count
    pnl_ratio = mean_win/-mean_lose

    if not (-1 in np.unique(sig) or 1 in np.unique(sig) or 0 in np.unique(sig)) and np.sum(np.where(trade>0,1,0))<5: 
        return {
        '年化收益':-1,
        '相关系数':-1,
        '卡玛比率':-1,
        '夏普比率':-1,
        '最大回撤': -1,
        '日胜率': -1,
        '次胜率': -1,
        '盈亏比': -1,
        '平均多头持仓天数':-1,
        '年均交易次数':-1,
            }

    return {
        '年化收益':round(ann_ret,4),
        '相关系数':round(corr,4),
        '卡玛比率':round(calmar_ratio,3),
        '夏普比率': round(sharpe,3),
        '最大回撤': round(max_dd,4),
        '日胜率': round(win_ratio_day,4),
        '次胜率': round(win_ratio_by_trade,4),
        '盈亏比': round(pnl_ratio,4),
        '平均多头持仓天数':round(avg_holding_day,2),
        '年均交易次数':round(avg_trade_count,2)
        }

def ma_diffusion(df_close,window):
    ma_series = df_close.rolling(window).mean()
    over_ma = np.where(df_close>ma_series,1,0)
    ma_ratio = over_ma.sum(axis=1)/df_close.count(axis=1)
    return ma_ratio

def roc_diffusion(df_close,window):
    roc_series = df_close.shift(window)
    over_roc = np.where(df_close>roc_series,1,0)
    roc_ratio = over_roc.sum(axis=1)/df_close.count(axis=1)
    return roc_ratio

def rsi_diffusion(df_close,window):
    rsi_series = df_close.apply(lambda x: talib.RSI(x,timeperiod = window))
    over_rsi = np.where(rsi_series>70,1,0)
    rsi_ratio = over_rsi.sum(axis=1)/df_close.count(axis=1)
    return rsi_ratio

def ma_crossover(n1,n2,series):
    return np.where(series.rolling(n1).mean()>=series.rolling(n2).mean(), 1, np.where(series.rolling(n1).mean()<=series.rolling(n2).mean(),-1,0))

def quantile_crossover(t_large,t_small,series):
    return np.where(series>=t_large, 1, np.where(series<=t_small,-1,0))

def quantile_crossover_reverse(t_large,t_small,series):
    return np.where(series>=t_large, -1, np.where(series<=t_small,1,0))

def make_bt_df(sig,col, origin_df):
    bt_df = origin_df.copy()
    bt_df[f'sig_{col}'] = sig
    bt_df[f'trade_{col}'] = bt_df[f'sig_{col}'].diff()
    return bt_df

def quantile_crossover_with_holding(macro_df,col,t_large,t_small,s_large,s_small):
    sig = []
    count = 0
    
    for i,row in macro_df.iterrows():
        
        if count > 0 :
            count -=1
            sig.append(temp_sig)
            continue
        else:
            if row[col]>t_large:
                count = 20
                temp_sig = s_large
                sig.append(temp_sig)
            elif row[col]<t_small:
                count = 20
                temp_sig = s_small
                sig.append(temp_sig)
            else:
                sig.append(0)
    return sig


def quantile_crossover_with_holding(series,t_large,t_small,s_large,s_small):
    sig = []
    count = 0
    
    for i in range(len(series)):
        
        if count > 0 :
            count -=1
            sig.append(temp_sig)
            continue
        else:
            if series.iloc[i]>t_large:
                count = 20
                temp_sig = s_large
                sig.append(temp_sig)
            elif series.iloc[i]<t_small:
                count = 20
                temp_sig = s_small
                sig.append(temp_sig)
            else:
                sig.append(0)
    return sig