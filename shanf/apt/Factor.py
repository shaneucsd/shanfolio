import pandas as pd
import numpy as np
import statsmodels.api as sm
from abc import ABC, abstractmethod
import dask.dataframe as dd
from dask.multiprocessing import get
# import scripts.backtest_utils as bt_utils

# BM_RET = bt_utils.get_benchmark_return('000300')

class Factor(ABC):

    """
    Abstract Class of Factor

    Key Functions:
        - Load data
        - Calculation
        - Preprocess
    
    Returns:
        fl: DtLoader
        factor: factor data, pd DataFrame, tradedays as index, tradeassets as columns
    """
    
    @classmethod
    def set_fl(cls, fl):
        cls.fl = fl
        
    @classmethod
    def set_basic_info(cls, basic_info:pd.DataFrame):
        cls.basic_info = basic_info
    
    

    def preprocess_ffill(self, df:pd.DataFrame) -> pd.DataFrame:
        # Preprocess: fillna by ffill
        df = df.replace({0:np.nan})
        df = df.ffill()
        return df
    
    def winsorize(self,df:pd.DataFrame)  -> pd.DataFrame:
        # Preprocess: Winsorize
        def winsorize_row(row):
            median = row.median()
            abs_median = (row - median).abs().median()
            upper_limit = median + 5 * abs_median
            lower_limit = median - 5 * abs_median
            return row.clip(upper=upper_limit, lower=lower_limit)
        
        return df.apply(winsorize_row, axis=1)
    
    def normalize(self,df:pd.DataFrame) -> pd.DataFrame:
        # Preprocess: Normalize
        return  df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    def fillna_by_sec_median(self,df):
        # Preprocess: fillna by sector median
        df = df.unstack().reset_index().copy()
        df = df.merge(Factor.basic_info[['symbol','swlevel1name']],how='left',on='symbol')
        sec_median = df.groupby(['tradedate','swlevel1name']).agg({0:lambda x: x.median(skipna=True)}).reset_index()
        df = pd.merge(df,sec_median,on=['tradedate','swlevel1name'],how='left')
        df['0_x'].fillna(df['0_y'],inplace=True)
        df.drop(['0_y'],axis=1,inplace=True)
        df = df.pivot(index='tradedate', columns='symbol', values='0_x')
        return df
    
            
    def neuturalize_sector_and_mktcap(self,df ):
        # Preprocess: neuturalize factor by sector and mktcap
        neu_factor = pd.DataFrame(index=df.index, columns=df.columns)
        
        mktcap = self.prevent_0(self.fl.negotiablemv)
        mktcap = np.log(mktcap) # 对数市值

        for day in self.factor.index:
            
            factor_data_day = self.factor.loc[day]
            market_cap_day = mktcap.loc[day]
            tradable_day = Factor.stock_pool.loc[day]
            
            valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day) & ~np.isnan(market_cap_day) & ~np.isinf(market_cap_day) & ~np.isnan(tradable_day)
            
            if valid_indices.sum()==0:
                neu_factor.loc[day, :] = np.nan
                
            else:    
                factor_data_valid = factor_data_day[valid_indices]
                market_cap_day_valid = market_cap_day[valid_indices]
                stock_sector_valid = self.sector_dummy[valid_indices]

                sector_with_mktcap = pd.concat([market_cap_day_valid,stock_sector_valid],axis=1)
                
                res = sm.OLS(factor_data_valid,sector_with_mktcap).fit()
                residual = res.resid
                neu_factor.loc[day, valid_indices] = residual
            
        return neu_factor
    
    def run(self):
        preprocess_funcs = [
            self.winsorize,
            self.normalize,
            self.neuturalize_sector_and_mktcap
            ]
        
        self.factor = self.calculate()
        for func in preprocess_funcs:
            self.factor = func(self.factor)
        return self.factor
    
    def _sync_days(self,_fin,tradedays):
        idx = _fin.index.union(tradedays)
        _fin = _fin.reindex(idx).ffill().fillna(0)
        _fin = _fin.loc[tradedays]
        return _fin
        
    def _rolling_decay(self,_x,_d,func = np.sum):
        _x = _x[~np.isnan(_x)]
        _d = _d[-len(_x):]
        return func(_x * _d)    
        
    def _dask_rolling_decay(self,df, window, decay_weights, min_periods,func=np.std, ncores=6):
        ddf = dd.from_pandas(df, npartitions=ncores)
        rolling_ddf = ddf.rolling(window=window, min_periods=min_periods)
        result_ddf = rolling_ddf.apply(lambda x: self._rolling_decay(x, decay_weights, func), raw=True)
        result_df = result_ddf.compute(scheduler='threads')
        return result_df
    
    def _regress(self,y, x, intercept=True, weight=1):
        if intercept:
            X = sm.add_constant(x)
        else:
            X = x
        model = sm.WLS(y, X, weights=weight)
        results = model.fit()
        return results.params[0], results.params[1]
    
    
class IndexComp_Factor(Factor):
    index_comp_df = None
    tradeassets = None 
    index_comp_dict = {}
    
    @classmethod
    def set_index_comp_df(cls,index_comp_df):
        cls.index_comp_df = index_comp_df
        
    @classmethod
    def set_index_comp_dict(cls,index_comp_dict):
        if cls.tradeassets is None:
            cls.index_comp_dict = index_comp_dict
        else:
            cls.index_comp_dict = {k:v for k,v in index_comp_dict.items() if k in cls.tradeassets}    
                        
    def calculate(**kwargs):
        pass
    
    