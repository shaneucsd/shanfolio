import sys
import os
import empyrical
import numpy as np
import pandas as pd
import configparser
import warnings
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.multiprocessing import get
from IPython.display import display
import tushare as ts 
config = configparser.ConfigParser()
config.read('/Users/apple/Documents/GitHub/shanfolio/config.ini')
data_path = config.get('path', 'data_path')
local_path = config.get('path', 'local_path')
sys.path.append(local_path)
ts.set_token(config.get('token', 'tushare'))


from shanf import apt 
from shanf.datakit.DtLoader import DtLoader 
from shanf.apt.Factor import Factor,IndexComp_Factor
from scripts import stats_utils as su
from scipy import stats
import datetime
pro = ts.pro_api()
warnings.filterwarnings('ignore')


def get_tradedays(start= '20060101',end = datetime.datetime.today().strftime('%Y%m%d')):
    tradedays = pro.trade_cal(exchange='',start_date=start,end_date=end).query('is_open==1')
    tradedays = pd.to_datetime(tradedays['cal_date']).sort_values().tolist()
    return tradedays

def get_tradeassets():
    return stock_fl.close.columns.tolist()

basic_info = pd.read_csv(os.path.join(data_path,'source/stock/basic_info/basic_info.csv'),encoding='gbk',parse_dates=['listdate','delistdate'])  
basic_info['symbol'] = basic_info['symbol'].astype(str).apply(lambda x: x.zfill(6))
basic_info = basic_info.drop_duplicates(subset=['symbol'],keep='first')


stock_fl = DtLoader(field_root=os.path.join(data_path, 'processed/stock'))
tradedays = get_tradedays(end='20230701')
tradeassets = get_tradeassets()
tradeassets = [ i for i in tradeassets if i in list(basic_info['symbol'].values) and i not in  ['301329', '301456', '301503', '833751']]
stock_fl.params.update({'tradeassets':tradeassets,'tradedays':tradedays})
stock_fl.reload()
stock_fl.set_default_days_assets()

stock_pool = pd.read_pickle('/Users/apple/Documents/GitHub/shanfolio/tests/stock_pool.pkl')

bm_price = pd.read_pickle(os.path.join(data_path,'processed/index/close.pkl'))['000300'].dropna().ffill()
bm_ret = np.log(bm_price).diff()
bm_ret = bm_ret.loc[tradedays]


Factor.set_fl(stock_fl)
Factor.set_basic_info(basic_info)
# Factor.stock_pool = stock_pool


sector_dummy = Factor.basic_info[['symbol','swlevel1name']]
sector_dummy = pd.get_dummies(sector_dummy[sector_dummy['symbol'].isin(tradeassets)].sort_values(by=['symbol'], key=lambda x: x.map(tradeassets.index)).set_index('symbol'),prefix='',prefix_sep='').astype(int)
Factor.sector_dummy = sector_dummy



class Valuation(Factor):
    def _sync_days(self,_fin,tradedays):
        idx = _fin.index.union(tradedays)
        _fin = _fin.reindex(idx).ffill().fillna(0)
        _fin = _fin.loc[tradedays]
        return _fin

    def BTOP(self):
        # mrq
        paresharrigh = self._sync_days(Factor.fl.paresharrigh,tradedays)
        prest = self._sync_days(Factor.fl.prest,tradedays)
        perbond = self._sync_days(Factor.fl.perbond,tradedays)
        totmktcap = Factor.fl.totmktcap.ffill()
        btop = (totmktcap*10000 / (paresharrigh-prest-perbond)).replace({np.inf:np.nan})
        return btop
    
factor = Valuation().BTOP()
factor = factor*stock_pool
factor.index.set_names(['date'],inplace=True)
factor = factor.stack()
factor.index.set_names(['date','asset'],inplace=True)
factor = factor.to_frame('factor_value')

close = Factor.fl.close.ffill()
prices=  close*stock_pool
prices.index.set_names(['date'], inplace=True)

import alphalens

forward_returns =  alphalens.utils.compute_forward_returns(factor, prices, periods=(1,5,10),)
                                            
factor_data = alphalens.utils.get_clean_factor(factor, forward_returns,quantiles=5)

alphalens.tears.create_full_tear_sheet(factor_data)