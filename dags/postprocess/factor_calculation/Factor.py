from abc import abstractmethod

import dask.dataframe as dd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import seaborn as sns 

from enums.index_estclass import IndexEstclass
from postprocess.datakit.DtLoader import DtLoader


class LazyProperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Factor:
    """
    Abstract Class of Factor

    Key Functions:
        - Load data
        - Calculation
        -
    
    Attributes:
        fl: DtLoader
        prefix: str, 'stock' or 'index'
    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        # prefix: 'stock','index'
        self.prefix = prefix
        self.index_estclass = index_estclass
        self.dl = DtLoader(prefix=prefix, index_estclass=index_estclass)

    @abstractmethod
    def calculate(self, field=None, time_period=None, time_period_ma=None, time_period2=None, weight_type=None,
                  comparison_type=None, components_pool=None, count_type=None, inflow=None, outflow=None, type=None,
                  method=None, fit=None):  # 有人有 有人没有
        pass

    def preprocess(self):
        pass

    def prevent_0(self, _x):
        _x = _x.replace({0: np.nan}).ffill()
        return _x

    def rolling_decay(self, _x, _d, func=np.sum):
        _x = _x[~np.isnan(_x)]
        _d = _d[-len(_x):]
        return func(_x * _d)

    def _dask_rolling_decay(self, _x, window, decay_weights, min_periods, func=np.std, ncores=6):
        ddf = dd.from_pandas(_x, npartitions=ncores)
        rolling_ddf = ddf.rolling(window=window, min_periods=min_periods)
        result_ddf = rolling_ddf.apply(lambda x: self.rolling_decay(x, decay_weights, func), raw=True)
        result_df = result_ddf.compute(scheduler='threads')
        return result_df

    def _regress(self, y, x, intercept=True, weight=1):
        if intercept:
            X = sm.add_constant(x)
        else:
            X = x
        model = sm.WLS(y, X, weights=weight)
        results = model.fit()
        return results.params[0], results.params[1]

    def orthogonalize_factor(self, y, x):
        # Make sure the tradedate is the same in both dataframes
        if not np.all(y.index == x.index):
            raise ValueError("The trade dates in liquidity and size data must match.")

        # Initialize a dataframe to store the residuals
        df = pd.DataFrame(index=y.index, columns=y.columns)

        # Loop over each trade date to run cross-sectional regression
        for tradedate in tqdm(df.index):

            # Extract liquidity and size data for the specific trade date
            _x = x[x.index == tradedate].values.flatten()
            _y = y[y.index == tradedate].values.flatten()

            _x = np.where(~np.isnan(_x) & ~np.isnan(_y), _x, np.nan)
            _y = np.where(~np.isnan(_y) & ~np.isnan(_y), _y, np.nan)

            _x = _x[~np.isnan(_x)]
            _y = _y[~np.isnan(_y)]

            if len(_x) > 0:
                _X = sm.add_constant(_x)  # Add a constant term for intercept
                model = sm.OLS(_y, _X)
                results = model.fit()
                alpha = results.params[0]
                beta = results.params[1]
            else:
                alpha, beta = np.nan, np.nan

            this_x = x[x.index == tradedate].values.flatten()
            this_y = y[y.index == tradedate].values.flatten()
            residuals = this_y - alpha - beta * this_x
            df.loc[tradedate, :] = residuals

        return df

    

class IndexCompFactor(Factor):
    index_comp_df = None
    tradeassets = None
    index_comp_dict = {}

    def __init__(self, prefix):
        super().__init__(prefix)

    @classmethod
    def set_index_comp_df(cls, index_comp_df):
        cls.index_comp_df = index_comp_df

    @classmethod
    def set_index_comp_dict(cls, index_comp_dict):
        if cls.tradeassets is None:
            cls.index_comp_dict = index_comp_dict
        else:
            cls.index_comp_dict = {k: v for k, v in index_comp_dict.items() if k in cls.tradeassets}

    def calculate(**kwargs):
        pass

