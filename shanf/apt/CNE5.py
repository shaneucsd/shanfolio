import os
import numpy as np
import pandas as pd
import configparser
from Factor import Factor
import statsmodels.api as sm
from pyfinance.utils import rolling_windows
from scripts import backtest_utils as bt_utils
config = configparser.ConfigParser()
config.read('config.ini')
data_path = config.get('path', 'data_path')
local_path = config.get('path', 'local_path')


bm_close = bt_utils.get_benchmark_close('000300')
bm_ret = bt_utils.get_benchmark_return('000300')


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

class Liquidity(Factor):
    """
    
    换手率：
        月均换手率
        季均换手率
        年均换手率
    
    """
    def calculate(self):
        return super().calculate() 
    
    @lazyproperty
    def factor(cls):
        return cls.STOM * 0.35 + cls.STOQ * 0.35 + cls.STOA * 0.3
    
    @lazyproperty
    def STOM(self):
        turnrate = Factor.fl.turnrate
        stom = turnrate.apply(lambda x: np.log(x.rolling(window=21, min_periods=14).sum()),axis=0)
        return stom
    
    @lazyproperty
    def STOQ(self):
        turnrate = Factor.fl.turnrate
        stoq = turnrate.apply(lambda x: np.log(x.rolling(window=63, min_periods=40).sum()),axis=0)
        return stoq
    
    @lazyproperty
    def STOA(self):
        turnrate = Factor.fl.turnrate
        stoa = turnrate.apply(lambda x: np.log(x.rolling(window=252, min_periods=120).sum()),axis=0)
        return stoa
    
## 111111111111111111111111111
class Momentum(Factor):

    @lazyproperty
    def factor(cls):
        return cls.RSTR * 1
    
    @lazyproperty
    def RSTR(self):        
        # Log return
        
        close = Factor.fl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret,axis=0)
        
        # lagged by 21 days
        lag = 21
        excess_ret = excess_ret.shift(lag)

        window = 504
        halflife = 126
        decay_weights = (0.5 ** (np.arange(window) / halflife))[::-1]
        
        rstr = self._dask_rolling_decay(excess_ret, window, decay_weights, min_periods=252, func=np.sum)
        return rstr
    
## 111111111111111111111111111
class Size():
    """
    
    对数市值
    
    """
    def LNCAP(self):
        lncap = np.log(Factor.fl.negotiablemv).ffill()
        return lncap
    
## 111111111111111111111111111
class NonLinearSize():
    
    """
    
    非线性市值
    
    """
    
    def _midcap_regress(self,mktcap):
        x = mktcap.dropna().values
        y = x**3
        alpha, beta = self._regress(y, x, intercept=True, weights=np.sqrt(x))
        resid = mktcap**3 - alpha - beta * mktcap
        return resid

    def MIDCAP(self):
        lncap = Size().LNCAP()
        midcap = lncap.apply(lambda x: self._midcap_regress(x) ,axis=1)
        return midcap
    
## 111111111111111111111111111
class ResidualVolatility(Factor):
    
    """
    
    残差波动率：
        日波动率
        累计收益范围
        残差波动率
        
    """
    
    @lazyproperty
    def DASTD(self):        
        
        close = Factor.fl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret,axis=0)
        
        # Parameters
        window = 252
        halflife = 42
        decay_weights = (0.5 ** (np.arange(window) / halflife))[::-1]

        # Run the Dask-based calculation
        dastd = self._dask_rolling_decay(excess_ret, window, decay_weights, min_periods=120, func=np.std)
        
        return dastd
    
    @lazyproperty
    def CMRA(self):

        close = Factor.fl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret,axis=0).fillna(0)
        
        zt = excess_ret.rolling(21).sum()

        for i in range(11):
            zt_max = np.maximum(zt.shift(i*21),zt.shift((i+1)*21))
            zt_min = np.minimum(zt.shift(i*21),zt.shift((i+1)*21))
            
        cmra = np.log(1+zt_max) - np.log(1+zt_min)
        return cmra
    
## 111111111111111111111111111
class BETA_HALPHA_HSIGMA(Factor):
    
    def _regress(self,y, x, intercept, weights):
        
        if intercept:
            X = sm.add_constant(x)
        else:
            X = x
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        return results.params[0], results.params[1]
    
    def beta_halpha_hsigma(self):
        
        close = Factor.fl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret,axis=0)

        window = 252
        halflife = 63
        decay_weights = 0.5 ** (np.arange(window) / halflife)

        alphas = pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
        betas = pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
        sigma = pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)

        _rolling_ret = rolling_windows(excess_ret.values,window)
        _rolling_bm_ret = rolling_windows(bm_ret.values,window)

        for i, (y,x) in enumerate(zip(_rolling_ret,_rolling_bm_ret)):
            
            if np.any(np.isnan(x)):
                alpha,beta,resid = np.nan,np.nan,np.nan
                
            else:
                # 获取回归系数，其中第一个系数为常数项，第二个系数即为beta因子
                x = sm.add_constant(x)
                model = sm.WLS(y, x, weights=decay_weights).fit()

                alpha = model.params[0]
                beta = model.params[1]
                resid = np.std(model.resid,axis=0)

            alphas.loc[excess_ret.index[i+window-1]] = alpha
            betas.loc[excess_ret.index[i+window-1]] = beta
            sigma.loc[excess_ret.index[i+window-1]] = resid
            
        return (betas, alphas, sigma)
   
# 000000000000000
class Leverage(Factor):
    """
    杠杆因子


    """
    
    def _idx_sync(self,df,index):
        _idx = df.index.union(index)
        df = df.reindex(_idx)
        df = df.sort_index().ffill().loc[index].fillna(0)
        return df
        
    @lazyproperty
    def MLEV(self):
        # 市场杠杆 
        # = (me+pe+ld) / me
        # me, totmktcap; pe, preferedequity; ld, totalnoncurrentliability;
        
        idx = Factor.fl.close.index
        
        me = Factor.fl.totmktcap
        pe = Factor.fl.prest
        ld = Factor.fl.totalnoncliab
        
        pe = self._idx_sync(pe,idx)
        ld = self._idx_sync(ld,idx)
        
        mlev = (me+pe+ld)/me
        
        return mlev
    
    @lazyproperty
    def BLEV(self):
        # 账面杠杆
        # (be+pe+ld)/be
        # 普通股账面价值,be ; 优先股账面价值, pe ; ld, 非流动性负债
        idx = Factor.fl.close.index 

        totalnoncliab = Factor.fl.totalnoncliab
        prest = Factor.fl.prest
        righaggr = Factor.fl.righaggr

        totalnoncliab = self._idx_sync(totalnoncliab,idx)
        prest = self._idx_sync(prest,idx)
        righaggr = self._idx_sync(righaggr,idx)

        ld = totalnoncliab
        pe = prest
        be = righaggr - pe

        blev = (be+pe+ld)/be
        
        return blev
    
    @lazyproperty
    def DTOA(self):
        # 负债总资产比
        # td / ta
        # ld, 非流动性负债; ta, 总资产

        idx = Factor.fl.close.index

        td = Factor.fl.totliab
        ta = Factor.fl.totasset
        
        td = self._idx_sync(td,idx)
        ta = self._idx_sync(ta,idx)
        
        dtoa = td/ta
        
        return dtoa
    
    


## 111111111111111111111111111
class Earnings(Factor):
    def _sync_days(self,_fin,tradedays):
        idx = _fin.index.union(tradedays)
        _fin = _fin.reindex(idx).ffill().fillna(0)
        _fin = _fin.loc[tradedays]
        return _fin
    
    def factor(self):
        return 0.68*self.EPFWD + 0.21*self.CP + 0.11*self.EP
    
    @lazyproperty
    def EPFWD(self):
        pe_fcst= Factor.fl.pe_fcst
        ep_fwd = 1/pe_fcst
        return ep_fwd
    
    @lazyproperty
    def EP(self):
        # mrq
        parenetp_ttm = self._sync_days(Factor.fl.parenetp_ttm,tradedays)
        totmktcap = Factor.fl.totmktcap.ffill()
        pe_ttm = totmktcap*10000/parenetp_ttm
        ep = 1/pe_ttm
        return ep
    
    @lazyproperty
    def CP(self):
        mananetr_ttm = self._sync_days(Factor.fl.mananetr_ttm,tradedays)
        totmktcap = Factor.fl.totmktcap.ffill()
        pc_ttm = totmktcap*10000/mananetr_ttm
        cp = 1/pc_ttm
        return pc_ttm
    
    
