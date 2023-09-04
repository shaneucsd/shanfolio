import numpy as np
from shanf.apt.Factor import Factor,IndexComp_Factor


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
    
class Momentum(Factor):
    
    @lazyproperty
    def RSTR(self):        
        # Log return
        log_ret = np.log(Factor.fl.close).diff()
        excess_ret = log_ret.sub(bm_ret,axis=0)
        
        # lagged by 21 days
        lag = 21
        log_ret = log_ret.shift(lag)

        # Calculate  the expoential decay weights
        window = 504
        halflife = 126
        decay_weights = 0.5 ** (np.arange(window) / halflife)
        
        # Apply the exponential decay weights to log returns
        weighted_returns = excess_ret.rolling(window=window, min_periods=1).apply(
            lambda x: np.sum(x * decay_weights), raw=True)
        
        return weighted_returns