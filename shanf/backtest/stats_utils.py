import pandas as pd 
import alphalens as al
from scipy import stats
from tqdm import tqdm 
import empyrical
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


"""

This file contains some functions that are used to calculate the statistics of the factors and the performance of the strategies.

Including:
@ ic_series: calculate the ic series of the factor
@ ic_stats: calculate the ic stats of the factor
@ performance_stats: calculate the performance stats of the strategy


"""

def ic_series(factor, returns):
    # factor : t * n
    # returns : t * n
    ic = factor.corrwith(returns,axis=1,method='spearman').dropna()
    assert len(ic)>0
    return ic

def ic_stats(factor, ic):
    # Here 
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean/ic_std
    ic_skew = ic.skew()
    t_value = ic_mean/ic_std*np.sqrt(len(factor))
    p_value = 2*(stats.t.sf(abs(t_value),factor.notna().count()[0]))
    ic_gt_2_ratio = len(ic[ic.abs()>0.02])/len(ic)
    ic_gt_0_ratio = len(ic[ic>0])/len(ic)

    return {
        'ic_mean':ic_mean,
        'ic_std':ic_std,
        'ic_ir':ic_ir,
        't_value':t_value,
        'p_value':p_value,
        'ic_gt_2_ratio':ic_gt_2_ratio,
        'ic_gt_0_ratio':ic_gt_0_ratio
    }
    

# 策略
def performance_stats(returns,market_index_return,period='daily',start=None,end=None):
    if start :
        returns = returns.loc[start:]
        market_index_return = market_index_return.loc[start:]
    if end : 
        returns = returns.loc[:end]
        market_index_return = market_index_return.loc[:end]
        
    # intersection index of returns and market_index_return
    returns = returns.loc[returns.index.intersection(market_index_return.index)]
    market_index_return = market_index_return.loc[market_index_return.index.intersection(returns.index)]
    annual_ret = empyrical.annual_return(returns,period=period)
    ann_pre_ret = annual_ret - empyrical.annual_return(market_index_return,period=period)
    info_ratio = empyrical.excess_sharpe(returns,market_index_return)
    annual_vol = empyrical.annual_volatility(returns,period=period)
    sharpe_ratio = empyrical.sharpe_ratio(returns,period=period)
    max_drawdown = empyrical.max_drawdown(returns)
    calmar_ratio = empyrical.calmar_ratio(returns,period=period)
    
    return {
        'annual_ret':annual_ret,
        'ann_pre_ret':ann_pre_ret,
        'annual_vol':annual_vol,
        'max_drawdown':max_drawdown,
        'sharpe_ratio':sharpe_ratio,
        'calmar_ratio':calmar_ratio,
        'info_ratio':info_ratio,
        'win_ratio':len(returns[(returns-market_index_return)>0])/len(returns)
        }

def get_ic_table_for_factors(factor_datas):
    ic_table = pd.DataFrame(index = factor_datas.keys(), columns = ['RankIC','ICIR','t-stat(IC)','p-value(IC)'])
    for key in tqdm(factor_datas.keys()):
        factor_data = factor_datas[key]
        ic = al.performance.factor_information_coefficient(factor_data)
        ic = ic_stats(ic)
        ic_table.loc[key,'RankIC'] = format(ic['1D']['IC Mean'],'.2%')
        ic_table.loc[key,'ICIR'] = format(ic['1D']['IC Mean']/ic['1D']['IC Std.'],'.2f')
        ic_table.loc[key,'t-stat(IC)'] = format(ic['1D']['t-stat(IC)'],'.2f')
        ic_table.loc[key,'p-value(IC)'] = format(ic['1D']['p-value(IC)'],'.2f')
    return ic_table

def get_performance_table_for_factors(factor_datas,market_index_return):
    # loop through all the factors and calculate the performance, also format into the df
    performance = pd.DataFrame(index = factor_datas.keys(), columns = ['ann_ret','ann_pre_ret','ann_vol','sharpe','max_drawdown','calmar'])

    for key in tqdm(factor_datas.keys()):
        factor_data = factor_datas[key]
        mean_ret_by_q, std_err_by_q = al.performance.mean_return_by_quantile(factor_data,by_date=True)
        long_short_return = (mean_ret_by_q.loc[5] - mean_ret_by_q.loc[1])['1D']
        performance.loc[key] = performance_stats(long_short_return,market_index_return)

    performance = performance.dropna()
    performance['ann_ret'] = performance['ann_ret'].apply(lambda x: format(x,'.2%'))
    performance['ann_pre_ret'] = performance['ann_pre_ret'].apply(lambda x: format(x,'.2%'))
    performance['ann_vol'] = performance['ann_vol'].apply(lambda x: format(x,'.2%'))
    performance['sharpe'] = performance['sharpe'].apply(lambda x: format(x,'.2f'))
    performance['max_drawdown'] = performance['max_drawdown'].apply(lambda x: format(x,'.2%'))
    performance['calmar'] = performance['calmar'].apply(lambda x: format(x,'.2f'))
    return performance

def get_combined_table_for_factors(factor_datas,market_index_return):
    ic_table = get_ic_table_for_factors(factor_datas)
    performance = get_performance_table_for_factors(factor_datas,market_index_return)
    combined_table = pd.concat([ic_table,performance],axis=1)

    return combined_table


def coskew(df, bias=False):
    v = df.values
    s1 = sigma = v.std(0, keepdims=True)
    means = v.mean(0, keepdims=True)

    # means is 1 x n (n is number of columns
    # this difference broacasts appropriately
    v1 = v - means

    s2 = sigma ** 2

    v2 = v1 ** 2

    m = v.shape[0]

    skew = pd.DataFrame(v2.T.dot(v1) / s2.T.dot(s1) / m, df.columns, df.columns)

    if not bias:
        skew *= ((m - 1) * m) ** .5 / (m - 2)

    return skew

def cokurt(df, bias=False, fisher=True, variant='middle'):
    v = df.values
    s1 = sigma = v.std(0, keepdims=True)
    means = v.mean(0, keepdims=True)

    # means is 1 x n (n is number of columns
    # this difference broacasts appropriately
    v1 = v - means

    s2 = sigma ** 2
    s3 = sigma ** 3

    v2 = v1 ** 2
    v3 = v1 ** 3

    m = v.shape[0]

    if variant in ['left', 'right']:
        kurt = pd.DataFrame(v3.T.dot(v1) / s3.T.dot(s1) / m, df.columns, df.columns)
        if variant == 'right':
            kurt = kurt.T
    elif variant == 'middle':
        kurt = pd.DataFrame(v2.T.dot(v2) / s2.T.dot(s2) / m, df.columns, df.columns)

    if not bias:
        kurt = kurt * (m ** 2 - 1) / (m - 2) / (m - 3) - 3 * (m - 1) ** 2 / (m - 2) / (m - 3)
    if not fisher:
        kurt += 3

    return kurt

def plot_distplot(series):
    sns.distplot(series)
    plt.show()
    print('mean: ',series.mean())
    print('std: ',series.std())
    print('skew: ',series.skew())
    print('kurt: ',series.kurt())
