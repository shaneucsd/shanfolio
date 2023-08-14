import sys
import os
sys.path.append('/Users/apple/Documents/Github/SHANFOLIO')
import time
import configparser
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from shanf.datakit.DtLoader import DtLoader
from shanf.apt.Factor import Factor,F_ret
import statsmodels.api as sm
config = configparser.ConfigParser()
config.read('config.ini')
data_path = config.get('path', 'data_path')


def winsorize_row(row):
    median = row.median()
    abs_median = (row - median).abs().median()
    
    upper_limit = median + 5 * abs_median
    lower_limit = median - 5 * abs_median
    
    return row.clip(upper=upper_limit, lower=lower_limit)

def normalize_row(row):
    z_score = (row - row.mean())/ row.std()
    return z_score

def neuturalize_mktcap(factor):
    neu_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    for day in factor.index:
        # 获取该交易日的因子数据和个股市值
        factor_data_day = factor.loc[day]
        market_cap_day = mktcap.loc[day]
        
        # 筛选有效值进行回归计算
        valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day) & ~np.isnan(market_cap_day) & ~np.isinf(market_cap_day)
        
        factor_data_valid = factor_data_day[valid_indices]
        market_cap_valid = market_cap_day[valid_indices]
        
        # 进行市值中性化处理
        model = sm.OLS(factor_data_valid, sm.add_constant(market_cap_valid))
        result = model.fit()
        residual = result.resid
        
        # 将计算结果赋值给 neu_factor
        neu_factor.loc[day, valid_indices] = residual
    return neu_factor


def neuturalize_sector(factor,basic_info):
    neu_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    stock_sector = pd.get_dummies(basic_info['SWLEVEL1NAME']).astype(int)
    
    for day in factor.index:
        
        factor_data_day = factor.loc[day]
        
        valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day)
        
        factor_data_valid = factor_data_day[valid_indices]
        stock_sector_valid = stock_sector[valid_indices]        

        res = sm.OLS(factor_data_valid,stock_sector_valid).fit()
        residual = res.resid
        neu_factor.loc[day, valid_indices] = residual
        
    return neu_factor
        

def neuturalize_sector_and_mktcap(factor,basic_info):
    neu_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    stock_sector = pd.get_dummies(basic_info['SWLEVEL1NAME']).astype(int)
    
    for day in factor.index:
        
        factor_data_day = factor.loc[day]
        market_cap_day = mktcap.loc[day]
        
        valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day) & ~np.isnan(market_cap_day) & ~np.isinf(market_cap_day)
        
        factor_data_valid = factor_data_day[valid_indices]
        market_cap_day_valid = market_cap_day[valid_indices]
        stock_sector_valid = stock_sector[valid_indices]

        sector_with_mktcap = pd.concat([market_cap_day_valid,stock_sector_valid],axis=1)
        

        res = sm.OLS(factor_data_valid,sector_with_mktcap).fit()
        residual = res.resid
        neu_factor.loc[day, valid_indices] = residual
        
    return neu_factor
        
START = '2010-01-01'
END = '2010-01-10'

stock_fl = DtLoader(field_root=os.path.join(data_path, 'processed/stock'))

basic_info = pd.read_csv(os.path.join(data_path,'source/stock/basic_info/basic_info.csv'),encoding='gbk')

basic_info['SYMBOL'] = basic_info['SYMBOL'].astype(str).apply(lambda x: x.zfill(6))

basic_info.set_index('SYMBOL',inplace=True)
basic_info = basic_info.loc[stock_fl.close.columns,:]
basic_info = basic_info[~basic_info.index.duplicated(keep='first')]



Factor.set_fl(stock_fl)
Factor.set_basic_info(basic_info)
Factor.fl.params.update({'start':START,'end':END})



x  = F_ret()
x.run()
print(x.factor)
# print(Factor.fl.factor)
