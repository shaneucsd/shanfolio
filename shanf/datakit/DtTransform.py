import pandas as pd
import numpy as np
import statsmodels.api as sm
from abc import ABC, abstractmethod
import dask.dataframe as dd
from dask.multiprocessing import get
# from shanf.datakit import DtLoader
# import scripts.backtest_utils as bt_utils


def preprocess(description):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.preprocess_func[func.__name__] = {'description': description}
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class DtTransform(ABC):
    
    
    properties = {
        'basic_info': None,
        'sector_dummy':None,
    }
    transforms = []
        
    @classmethod
    def set_properties(cls, properties:dict):
        cls.set_properties = properties

    
    def winsorize(self,df:pd.DataFrame)  -> pd.DataFrame:
        def winsorize_row(row):
            median = row.median()
            abs_median = (row - median).abs().median()
            upper_limit = median + 5 * abs_median
            lower_limit = median - 5 * abs_median
            return row.clip(upper=upper_limit, lower=lower_limit)
        
        return df.apply(winsorize_row, axis=1)
    
    def normalize(self,df:pd.DataFrame) -> pd.DataFrame:
        return  df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def replace_zero(self,df:pd.DataFrame) -> pd.DataFrame:
        return df.replace({0:np.nan})
    
    def fillna_ffill(self,df:pd.DataFrame) -> pd.DataFrame:
        return df.ffill()
    
    def fillna_by_sec_median(self, df:pd.DataFrame,properties:dict) -> pd.DataFrame:        
        """
        properties, the entity properties, including basic_info, mktcap, and etc.
        """
        basic_info = properties.get('basic_info')
        
        df = df.unstack().reset_index().copy()
        df = df.merge(basic_info[['symbol','swlevel1name']],how='left',on='symbol')
        sec_median = df.groupby(['tradedate','swlevel1name']).agg({0:lambda x: x.median(skipna=True)}).reset_index()
        df = pd.merge(df,sec_median,on=['tradedate','swlevel1name'],how='left')
        df['0_x'].fillna(df['0_y'],inplace=True)
        df.drop(['0_y'],axis=1,inplace=True)
        df = df.pivot(index='tradedate', columns='symbol', values='0_x')
        return df
    
    def neuturalize_sector_and_mktcap(self, df:pd.DataFrame,properties:dict,sector_neu = True, mktcap_neu = True) -> pd.DataFrame:        
        # Actually a Regress of factor on market cap and sector dummy
        negotiablemv = properties.get('negotiablemv') # [tradedate, symbol]
        sector_dummy = properties.get('sector_dummy') # [symbol, industry dummy]
        
        
        neu_factor = pd.DataFrame(index=df.index, columns=df.columns)
        
        for day in df.index:
            
            factor_data_day = df.loc[day]
            market_cap_day = negotiablemv.loc[day]            
            valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day) & ~np.isnan(market_cap_day) & ~np.isinf(market_cap_day)
            
            if valid_indices.sum()==0:
                neu_factor.loc[day, :] = np.nan
                
            else:    
                factor_data_valid = factor_data_day[valid_indices]
                market_cap_day_valid = market_cap_day[valid_indices]
                stock_sector_valid = sector_dummy[valid_indices]

            if sector_neu and mktcap_neu:
                neuturalizer = pd.concat([market_cap_day_valid,stock_sector_valid],axis=1)
            elif sector_neu and not mktcap_neu:
                neuturalizer = stock_sector_valid
            elif not sector_neu and mktcap_neu:
                neuturalizer = market_cap_day_valid
            
            # Regress, get residuals as neutralized factor
            neu_factor.loc[day, valid_indices] = sm.OLS(factor_data_valid,neuturalizer).fit().resid
        
        return neu_factor
    
    def run(self, df, transforms, **kwargs):
        """
        Example:
            transforms = [DtTransform.winsorize, DtTransform.normalize, DtTransform.neuturalize_sector_and_mktcap]
            # Method 1
            DtTransform.run(df, transforms, proerties = {'basic_info':basic_info,'negotiablemv':negotiablemv,'sector_dummy':sector_dummy},
                        neuturalize_sector_and_mktcap = {'sector_neu':True, 'mktcap_neu':True})
            # Method 2 
            kwargs = {'neuturalize_sector_and_mktcap':{'sector_neu':True, 'mktcap_neu':True},'properties':{'basic_info':basic_info,'negotiablemv':negotiablemv,'sector_dummy':sector_dummy}}}      
                
            
        """
        
        for func in transforms:
            df = func(df, **kwargs.get(func.__name__, {}))
        return df

    