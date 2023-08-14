import pandas as pd
import numpy as np
import statsmodels.api as sm
from abc import ABC, abstractmethod


class Factor(ABC):

    """
    Abstract Class of Factor

    Key Functions:
        - Calculation
        - Preprocess
    
    Returns:
        fl: DtLoader
        factor: factor data, pd DataFrame, tradedays as index, tradeassets as columns
    """
    
    @abstractmethod
    def calculate(self):
        pass
    
    @classmethod
    def set_fl(cls,fl):
        cls.fl = fl
        
    @classmethod
    def set_basic_info(cls,basic_info):
        cls.basic_info = basic_info
    
    def prevent_0(self,factor):
        factor = factor.replace({0:np.nan})
        return factor
    
    def preprocess(func):
        def wrapper(self, *args, **kwargs):
            self.factor = func(self)
            return self.factor
        return wrapper

    @preprocess
    def winsorize(self):
        def winsorize_row(row):
            median = row.median()
            abs_median = (row - median).abs().median()
            
            upper_limit = median + 5 * abs_median
            lower_limit = median - 5 * abs_median
            
            return row.clip(upper=upper_limit, lower=lower_limit)
        self.factor = self.factor.apply(winsorize_row, axis=1)
        return self.factor
    
    @preprocess
    def normalize(self):
        def normalize_row(row):
            z_score = (row - row.mean())/ row.std()
            return z_score
        self.factor = self.factor.apply(normalize_row,axis=1)
        return self.factor

    
    @preprocess
    def fillna(self):
        self.factor = self.factor.fillna(0)
        return self.factor
        
    @preprocess
    def neuturalize_mktcap(self):
        mktcap = self.prevent_0(self.fl.totmktcap)
        mktcap = np.log(mktcap) # 对数市值
        print('Done')
        
        for day in self.factor.index:
            # 获取该交易日的因子数据和个股市值
            factor_data_day = self.factor.loc[day]
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
            self.factor.loc[day, valid_indices] = residual
            
        return self.factor
    
    
    @preprocess
    def neuturalize_sector(self):
        neu_factor = pd.DataFrame(index=self.factor.index, columns=self.factor.columns)
        stock_sector = pd.get_dummies(self.basic_info['SWLEVEL1NAME']).astype(int)
        
        for day in self.factor.index:
            
            factor_data_day = self.factor.loc[day]
            
            valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day)
            
            factor_data_valid = factor_data_day[valid_indices]
            stock_sector_valid = stock_sector[valid_indices]        

            res = sm.OLS(factor_data_valid,stock_sector_valid).fit()
            residual = res.resid
            neu_factor.loc[day, valid_indices] = residual
            
        return neu_factor
            
    @preprocess
    def neuturalize_sector_and_mktcap(self):
        neu_factor = pd.DataFrame(index=self.factor.index, columns=self.factor.columns)
        stock_sector = pd.get_dummies(self.basic_info['SWLEVEL1NAME']).astype(int)
        mktcap = self.prevent_0(self.fl.totmktcap)
        mktcap = np.log(mktcap) # 对数市值
        
        for day in self.factor.index:
            
            factor_data_day = self.factor.loc[day]
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

    def run(self):
        preprocess_funcs = [
            self.prevent_0,
            self.winsorize,
            self.normalize,
            self.fillna,
            self.neuturalize_mktcap
            ]
        
        self.factor = self.calculate()
        for func in preprocess_funcs:
            self.factor = func(self.factor)
        return self.factor
    
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