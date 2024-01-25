import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from enums.index_estclass import IndexEstclass
from utils.file_utils import load_object_from_pickle

from postprocess.datakit.DtTransform import DtTransform
from postprocess.factor_calculation.Factor import Factor
from utils import backtest_utils as bt_utils

# trans ,这里是一个DtTransform对象,用于处理因子数据
# cne5为股票因子, prefix='stock'
trans = DtTransform('stock')
kwargs = {'neuturalize_sector_and_mktcap':{'sector_neu':False, 'mktcap_neu':False}}
transforms = [trans.winsorize, trans.normalize]

def linreg(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    y_pred = X @ beta
    residuals = y - y_pred
    return residuals
"""
Barra:
    SIZE 市值
    BETA 贝塔
    MOMENTUM 动量
    RESIDUALVOLATILITY 残差波动率
    NONLINEARSIZE 非线性市值
    VALUATION 估值
    LIQUIDITY 流动性
    EARNINGS 盈利
    GROWTH 成长
    LEVERAGE 杠杆
    DIVIDENDYIELD 股息率
"""

#################################### SIZE ####################################

class LNCAP(Factor):
    def calculate(self):
        mv = self.dl.negotiablemv
        mv = self.prevent_0(mv)
        lncap = np.log(mv)
        return lncap


class SIZE(Factor):
    def calculate(self):
        size = LNCAP(self.prefix).calculate()
        size = trans.run(size, transforms=transforms, **kwargs)
        return size


#################################### BETA ####################################

class BETA_HALPHA_HSIGMA(Factor):
    """
    Bug Record: Tradedays bm_close会自动存入非交易日,导致bm_ret的index不是交易日,从而导致个股的值总有缺失,
    
    """

    def calculate(self, return_hsigma=False,bm='000985'):

        def my_rolling_windows(arr, window):
            shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
            strides = (arr.strides[0],) + arr.strides
            return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

        close = self.dl.close.replace({0: np.nan}).ffill()
        log_ret = np.log(close).diff()
        
        bm_ret = bt_utils.get_benchmark_return(bm)
        
        excess_ret = log_ret.sub(bm_ret, axis=0)

        window = 252
        halflife = 63
        decay_weights = 0.5 ** (np.arange(window) / halflife)

        # TODO:
        # 应该改为tradedays和tradeassets
        shape_0 = excess_ret.shape[0]
        shape_1 = excess_ret.shape[1]
        index = excess_ret.index
        columns = excess_ret.columns

        halpha = np.zeros((shape_0, shape_1))
        beta = np.zeros((shape_0, shape_1))
        hsigma = np.zeros((shape_0, shape_1))

        # fill will nan of the first 'window' rows
        halpha[:window, :] = np.nan
        beta[:window, :] = np.nan

        _rolling_ret = my_rolling_windows(excess_ret.values, window)
        bm_ret = bt_utils.get_benchmark_return('000985')
        _rolling_bm_ret = my_rolling_windows(bm_ret.values, window)

        for i, (y, x) in enumerate(zip(_rolling_ret, _rolling_bm_ret)):
            if not np.any(np.isnan(x)):
                x = sm.add_constant(x)  # Design matrix
                Y = y  # Response matrix

                # Calculate beta coefficients using weighted least squares
                W = np.diag(decay_weights)

                XTX_inv = np.linalg.inv(x.T @ W @ x)
                beta_matrix = XTX_inv @ x.T @ W @ Y

                # Directly assign to pre-allocated arrays
                halpha[i + window - 1, :] = beta_matrix[0]
                beta[i + window - 1, :] = beta_matrix[1]

                if return_hsigma:
                    residual_matrix = Y - x @ beta_matrix
                    residual_std = np.std(residual_matrix, axis=0)
                    hsigma[i + window - 1, :] = residual_std
                else:
                    hsigma[i + window - 1, :] = np.nan

            else:
                halpha[i + window - 1, :] = np.nan
                beta[i + window - 1, :] = np.nan
                hsigma[i + window - 1, :] = np.nan

        halpha = pd.DataFrame(halpha, index=index, columns=columns)
        beta = pd.DataFrame(beta, index=index, columns=columns)
        hsigma = pd.DataFrame(hsigma, index=index, columns=columns)
        return beta, halpha, hsigma


class BETA(Factor):
    def calculate(self):
        beta, halpha, hsigma = BETA_HALPHA_HSIGMA(self.prefix).calculate(return_hsigma=False)
        beta = trans.run(beta, transforms=transforms, **kwargs)
        return beta


#################################### MOMENTUM ####################################

class RSTR(Factor):
    def calculate(self,bm='000985'):
        bm_ret = bt_utils.get_benchmark_return(bm)
        
        close = self.dl.close.ffill()
        log_ret = np.log(close).diff()
        bm_ret = bt_utils.get_benchmark_return('000985')
        excess_ret = log_ret.sub(bm_ret, axis=0)

        # lagged by 21 days
        lag = 21
        excess_ret = excess_ret.shift(lag)

        window = 504
        halflife = 126
        decay_weights = (0.5 ** (np.arange(window) / halflife))[::-1]

        rstr = self._dask_rolling_decay(excess_ret, window, decay_weights, min_periods=252, func=np.sum)
        return rstr


class Momentum(Factor):
    def calculate(self,bm = '000985'):
        rstr = RSTR(self.prefix).calculate(bm)
        momentum = trans.run(rstr, transforms=transforms, **kwargs)
        return momentum


#################################### ResidualVolatility ####################################

class DASTD(Factor):
    def calculate(self,bm = '000985'):
        bm_ret = bt_utils.get_benchmark_return(bm)
        close = self.dl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret, axis=0)

        # Parameters
        window = 252
        halflife = 42
        decay_weights = (0.5 ** (np.arange(window) / halflife))[::-1]

        # Run the Dask-based calculation
        dastd = self._dask_rolling_decay(excess_ret, window, decay_weights, min_periods=120, func=np.std)

        return dastd


class CMRA(Factor):
    def calculate(self,bm = '000985'):
        
        bm_ret = bt_utils.get_benchmark_return(bm)
        close = self.dl.close.ffill()
        log_ret = np.log(close).diff()
        excess_ret = log_ret.sub(bm_ret, axis=0).fillna(0)

        zt = excess_ret.rolling(21).sum()
        zt_max, zt_min = zt.copy(), zt.copy()

        for i in range(1, 12):
            zt_max = np.maximum(zt_max, zt.shift(i * 21))
            zt_min = np.minimum(zt_min, zt.shift(i * 21))

        cmra = np.log(1 + zt_max) - np.log(1 + zt_min)
        return cmra


class HSIGMA(Factor):
    def calculate(self):
        beta, halpha, hsigma = BETA_HALPHA_HSIGMA(self.prefix).calculate(return_hsigma=True)
        return hsigma


class ResidualVolatility(Factor):
    def calculate(self):
        dastd = DASTD(self.prefix).calculate()
        cmra = CMRA(self.prefix).calculate()
        hsigma = HSIGMA(self.prefix).calculate()
        rv = 0.74 * dastd + 0.16 * cmra + 0.10 * hsigma
        rv = trans.run(rv, transforms=transforms, **kwargs)
        return rv


#################################### Non Linear Size ####################################

class NonLinearSize(Factor):
    """
    
    非线性市值
    
    """
    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_pool = load_object_from_pickle('stock_pool_all')

    def calculate(self):
        size = SIZE(self.prefix).calculate()
        
        # size normalize
        _size = size*self.stock_pool
        _size = np.float32(_size)
        mean, std = np.nanmean(_size, axis=0), np.nanstd(_size, axis=0)
        clip_multiplier = 3
        upper, lower = mean + (clip_multiplier * std), mean - (clip_multiplier * std)
        _size = np.clip(_size, lower, upper)
        _size = (_size - np.nanmean(_size,axis=1)[:,None])/np.nanstd(_size,axis=1)[:,None]
        _size[np.isnan(_size)] = 0
        
        # reg
        non_linear_size = np.zeros_like(_size)

        _size_cube = -_size**2

        for i in range(len(_size)):
            non_linear_size[i,:] = linreg(_size[i,:],_size_cube[i,:])
            
        mean, std = np.nanmean(non_linear_size, axis=0), np.nanstd(non_linear_size, axis=0)
        clip_multiplier = 3
        upper, lower = mean + (clip_multiplier * std), mean - (clip_multiplier * std)
        non_linear_size = np.clip(non_linear_size, lower, upper)
        non_linear_size = (
            non_linear_size - np.nanmean(non_linear_size, axis=1)[:, None]
        ) / np.nanstd(non_linear_size, axis=1)[:, None]

        non_linear_size = pd.DataFrame(
            non_linear_size, index=size.index, columns=size.columns
        )

        non_linear_size = trans.run(
            non_linear_size.copy(), transforms=transforms, **kwargs
        )

        return non_linear_size

############################ Valuation  #############################################

class BTOP(Factor):

    def calculate(self):
        paresharrigh = self.dl.get_financial_field_df('paresharrigh')
        prest = self.dl.get_financial_field_df('prest')
        perbond = self.dl.get_financial_field_df('perbond')

        prest = prest.fillna(0)
        perbond = perbond.fillna(0)
        totmktcap = self.dl.totmktcap.ffill()
        btop = (paresharrigh - prest - perbond) / (totmktcap * 10000)
        return btop


class Valuation(Factor):
    def calculate(self):
        valuation =  BTOP(self.prefix).calculate()
        valuation = trans.run(valuation,transforms=transforms,**kwargs)
        return valuation

#################################### Liquidity ####################################
class STOM(Factor):
    def calculate(self):
        turnrate = self.dl.turnrate
        turnrate = self.prevent_0(turnrate)
        stom = turnrate.apply(lambda x: np.log(x.rolling(window=21).sum()), axis=0)
        return stom


class STOQ(Factor):
    def calculate(self):
        # From MSCI
        stom = STOM(self.prefix).calculate()
        i = 3
        stoq = pd.DataFrame(0, index=stom.index, columns=stom.columns)
        for i in range(i):
            stoq = stoq + np.exp(stom.shift(21 * i))
        stoq = np.log(stoq / i)
        return stoq

        # # From Github
        # turnrate = self.fl.turnrate
        # turnrate = self.prevent_0(turnrate)
        # stoq = turnrate.apply(lambda x: np.log(x.rolling(window=63).sum()), axis=0)
        # return stoq


class STOA(Factor):
    def calculate(self):
        # From MSCI
        stom = STOM(self.prefix).calculate()
        i = 12
        stoa = pd.DataFrame(0, index=stom.index, columns=stom.columns)
        for i in range(i):
            stoa = stoa + np.exp(stom.shift(21 * i))
        stoa = np.log(stoa / i)
        return stoa

        # # From Github
        # turnrate = self.fl.turnrate
        # turnrate = self.prevent_0(turnrate)
        # stoa = turnrate.apply(lambda x: np.log(x.rolling(window=252).sum()), axis=0)
        # return stoa


class Liquidity(Factor):
    def calculate(self):
        size = SIZE(prefix=self.prefix).calculate()
        stom = STOM(prefix=self.prefix).calculate()
        stoq = STOQ(prefix=self.prefix).calculate()
        stoa = STOQ(prefix=self.prefix).calculate()
        liquidity = stom * 0.35 + stoq * 0.35 + stoa * 0.3
        liquidity = self.orthogonalize_factor(liquidity, size)
        liquidity = trans.run(liquidity, transforms=transforms, **kwargs)
        return liquidity


############################ Earnings #############################################

class EPFWD(Factor):
    def calculate(self):
        epfwd = 1 / self.dl.pe_fcst
        return epfwd


class EP(Factor):
    # PE TTM is used
    def calculate(self):
        parenetp_ttm = self.dl.get_financial_field_df('parenetp_ttm')
        totmktcap = self.dl.totmktcap.ffill()
        ep = parenetp_ttm / (totmktcap * 10000)
        return ep


class CP(Factor):
    def calculate(self):
        mananetr_ttm = self.dl.get_financial_field_df('mananetr_ttm')
        totmktcap = self.dl.totmktcap.ffill()
        cp = mananetr_ttm / (totmktcap * 10000)
        return cp



class Earnings(Factor):
    def calculate(self):
        epfwd = EPFWD(self.prefix).calculate()
        ep = EP(self.prefix).calculate()
        cp = CP(self.prefix).calculate()
        earnings = 0.68 * epfwd + 0.21 * cp + 0.11 * ep
        earnings = trans.run(earnings.copy(), transforms=transforms, **kwargs)
        return earnings


############################ Growth ##############################

# Customize
# 经营性现金流同比增长率
# 净资产同比增长率
# 营业收入同比增长率
# 净利润同比增长率

# 有些公司有些财报就是不公布营业收入,比如600670,所以这里的营业收入同比增长率就是inf
# 大多数是退市公司
# 用空值替代,大概有42家公司有这种情况
class FinFieldTimeGrowth(Factor):
    """
    
    计算财务数据的增长率,
        - 过去n个财年的财务数据,对时间做线性回归,得到斜率,即为增长率。斜率需要除以财务数据的平均值,得到相对增长率。
        - 历史不满n个财年的用最早一期的数据替代。
        - 不直接使用n个财年的财务数据，而是用ttm数据的增长率。
        
    Args:
        Factor (_type_): _description_
    """

    def calculate(self, field):
        # 这里的财务数据都是ttm的，而且已经除以了股票总股本， xxx per share
        _field = self.dl.get_financial_field_df(field) / self.dl.stk_share / 10000
        _field = _field.replace({0: np.nan})
        _field.bfill(inplace=True)
        _field = _field

        # past 3 years
        _field_1y = _field.shift(252)
        _field_2y = _field.shift(252*2)

        # to numpy
        _field = _field.values
        _field_1y = _field_1y.values
        _field_2y = _field_2y.values

        betas = np.full(_field.shape,np.nan)

        # regression
        for i in range(len(_field)):
            
            p_list = [
                _field_2y[i],
                _field_1y[i],
                _field[i],
            ]
            p_list = np.vstack(p_list)
            x = np.array([i+1 for i in range(len(p_list))])
            X = np.column_stack([np.ones(len(x)), x])
            betas[i] = (np.linalg.lstsq(X, p_list, rcond=None)[0][1])/p_list.mean(0)

        # Create a DataFrame to store betas
        betas = pd.DataFrame(betas, index=self.dl.tradedays, columns=self.dl.tradeassets)
        
        return betas


class EGROW(Factor):
    def calculate(self):
        inco_growth = FinFieldTimeGrowth(self.prefix).calculate("bizinco_ttm")
        return inco_growth


class NPGROW(Factor):
    def calculate(self):
        net_profit_growth = FinFieldTimeGrowth(self.prefix).calculate('parenetp_ttm')
        return net_profit_growth

class EGRSF(Factor):
    # 短期盈利增长率
    # 次年forecast / ttm
    def calculate(self):
        egrsf = self.dl.net_revenue_fcst_2.div(self.dl.get_financial_field_df('parenetp_ttm')/10000)-1
        return egrsf

class EGRLF(Factor):
    # 长期盈利增长率
    # 未来3年forecast / ttm
    def calculate(self):
        egrlf = self.dl.net_revenue_fcst.div(self.dl.get_financial_field_df('parenetp_ttm')/10000)-1
        return egrlf


class Growth(Factor):
    # 0.18 * 预期长期盈利增长率 + 0.11 * 预期短期盈利增长率 + 0.24 * 5年盈利增长率 + 0.47 * 5年营业收入增长率
    def calculate(self):
        egrsf = EGRSF(self.prefix).calculate()
        egrlf = EGRLF(self.prefix).calculate()
        egrow = EGROW(self.prefix).calculate()
        npgrow = NPGROW(self.prefix).calculate()
        growth = 0.18 * egrlf + 0.11 * egrsf + 0.24 * npgrow + 0.47 * egrow
        growth = trans.run(growth, transforms=transforms, **kwargs)
        return growth


############################ Leverage ##############################

class MLEV(Factor):
    # 市场杠杆
    def calculate(self):
        pe = self.dl.get_financial_field_df('prest') / 10000
        ld = self.dl.get_financial_field_df('totalnoncliab') / 10000
        me = self.dl.totmktcap.ffill()

        pe = pe.fillna(0)
        ld = ld.fillna(0)
        mlev = (me + pe + ld) / me
        return mlev


class BLEV(Factor):
    # 账面杠杆
    def calculate(self):
        totalnoncliab = self.dl.get_financial_field_df('totalnoncliab') / 10000
        pe = self.dl.get_financial_field_df('prest') / 10000
        righaggr = self.dl.get_financial_field_df('righaggr') / 1000

        pe = pe.fillna(0)
        righaggr = righaggr.fillna(0)
        ld = totalnoncliab
        be = righaggr - pe
        blev = (be + pe + ld) / be
        return blev


class DTOA(Factor):
    def calculate(self):
        td = self.dl.get_financial_field_df('totliab') / 10000
        ta = self.dl.get_financial_field_df('totasset') / 10000
        td = td.fillna(0)
        ta = ta.fillna(0)
        dtoa = td / ta
        return dtoa


class Leverage(Factor):
    def calculate(self):
        mlev = MLEV(self.prefix).calculate()
        blev = BLEV(self.prefix).calculate()
        dtoa = DTOA(self.prefix).calculate()
        leverage = mlev * 0.5 + blev * 0.3 + dtoa * 0.2
        leverage = trans.run(leverage, transforms=transforms, **kwargs)
        return leverage



############################ Dividend Yield ##############################

class DY(Factor):
    def calculate(self):
        dy = self.dl.dy.ffill()
        dy = trans.run(dy, transforms=transforms, **kwargs)
        return dy

############################ ROE ##############################

class ROE(Factor):
    def calculate(self,is_ttm = False,is_avg=True):
        profit = self.dl.get_financial_field_df('parenetp_ttm') if is_ttm else self.dl.get_financial_field_df('parenetp')
        equity = self.dl.get_financial_field_df('paresharrigh') if is_avg == False else \
            ( self.dl.get_financial_field_df('paresharrigh') + self.dl.get_financial_field_df('paresharrigh_1') + self.dl.get_financial_field_df('paresharrigh_2') + self.dl.get_financial_field_df('paresharrigh_3'))/4
        roe = profit / equity
        return roe
    
class ROEYoY(Factor):
    def calculate(self,time_period = 252, is_ttm = False, is_avg = True):
        profit = self.dl.get_financial_field_df('parenetp_ttm') if is_ttm else self.dl.get_financial_field_df('parenetp')
        equity = self.dl.get_financial_field_df('paresharrigh') if is_avg == False else \
            ( self.dl.get_financial_field_df('paresharrigh') + self.dl.get_financial_field_df('paresharrigh_1') + self.dl.get_financial_field_df('paresharrigh_2') + self.dl.get_financial_field_df('paresharrigh_3'))/4
        roe = profit / equity
        roe_yoy = roe.pct_change(periods=time_period)
        return roe_yoy
    
class ROEYoYReal(Factor):
    def calculate(self,time_period = 252, is_ttm = False, is_avg = True):
        profit = self.dl.get_financial_field_df('parenetp_ttm') if is_ttm else self.dl.get_financial_field_df('parenetp')
        equity = self.dl.get_financial_field_df('paresharrigh') if is_avg == False else \
            ( self.dl.get_financial_field_df('paresharrigh') + self.dl.get_financial_field_df('paresharrigh_1') + self.dl.get_financial_field_df('paresharrigh_2') + self.dl.get_financial_field_df('paresharrigh_3'))/4
        equity[equity<=0] = np.nan
        roe = np.log(profit / equity)
        roe_yoy = roe.diff(periods=time_period)
        return roe_yoy
        
class ROA(Factor):
    def calculate(self, is_ttm=False, is_avg=True):
        profit = self.dl.get_financial_field_df('parenetp_ttm') if is_ttm else self.dl.get_financial_field_df('parenetp')
        equity = self.dl.get_financial_field_df('totasset')if is_avg == False else \
            ( self.dl.get_financial_field_df('totasset') + self.dl.get_financial_field_df('totasset_1') + self.dl.get_financial_field_df('totasset_2') + self.dl.get_financial_field_df('totasset_3'))/4
        roa = profit / equity
        return roa
    
class ROAYoY(Factor):
    def calculate(self,time_period = 252, is_ttm=False, is_avg=True):
        profit = self.dl.get_financial_field_df('parenetp_ttm') if is_ttm else self.dl.get_financial_field_df('parenetp')        
        equity = self.dl.get_financial_field_df('totasset')if is_avg == False else \
            ( self.dl.get_financial_field_df('totasset') + self.dl.get_financial_field_df('totasset_1') + self.dl.get_financial_field_df('totasset_2') + self.dl.get_financial_field_df('totasset_3'))/4
        roa = profit / equity
        roa_yoy = roa.pct_change(periods=time_period)
        return roa_yoy
    
class MarginRate(Factor):
    def calculate(self):
        bizinco = self.dl.get_financial_field_df('bizinco')
        bizcost = self.dl.get_financial_field_df('bizcost')
        margin = (bizinco-bizcost) / bizinco
        return margin
    
class MainMarginRate(Factor):
    def calculate(self):
        mainbizinco = self.dl.get_financial_field_df('mainbizinco')
        mainbizcost = self.dl.get_financial_field_df('mainbizcost')
        mainmargin = (mainbizinco-mainbizcost) / mainbizinco
        return mainmargin

class TurnoverRate(Factor):
    def calculate(self, is_ttm=False, is_avg=True):
        bizinco = self.dl.get_financial_field_df('bizinco_ttm') if is_ttm else self.dl.get_financial_field_df('bizinco')
        totasset = self.dl.get_financial_field_df('totasset')if is_avg == False else \
            ( self.dl.get_financial_field_df('totasset') + self.dl.get_financial_field_df('totasset_1') + self.dl.get_financial_field_df('totasset_2') + self.dl.get_financial_field_df('totasset_3'))/4
        turnover = bizinco / totasset
        return turnover

class DebtToAssetRate(Factor):
    """负债比率

    Args:
        Factor (_type_): totliab/totasset
    """    
    def calculate(self):
        totliab = self.dl.get_financial_field_df('totliab')
        totasset = self.dl.get_financial_field_df('totasset')
        debt_to_asset = totliab / totasset
        return debt_to_asset
    
class LiquidityDebtToAssetRate(Factor):
    """流动比率

    Args:
        Factor (_type_): 流动资产/流动负债
    """    
    def calculate(self):
        totcurrliab = self.dl.get_financial_field_df('totalcurrliab')
        totcurrasset = self.dl.get_financial_field_df('totcurrasset')
        liquidity_debt_to_asset = totcurrasset / totcurrliab 
        return liquidity_debt_to_asset


class Fscore(Factor):
    """
        ROA(Return on Assets): 净利润除以总资产
        ΔROA(Change in ROA): 当前期ROA减去上一期ROA
        Δ CFO(Cash Flow from Operations): 经营活动现金流量净额 [ΔCFO(Change in CFO): 当前期CFO减去上一期CFO]
        Δ ACCRUALS(Accruals): 当前期净利润减去CFO
        Δ MARGIN(Change in Gross Margin): 当前期毛利率减去上一期毛利率
        Δ TURN(Change in Asset Turnover): 当前期总资产周转率减去上一期总资产周转率
        Δ LEVERAGE(Change in Leverage): 当前期负债比率减去上一期负债比率 [LEVERAGE(Leverage): 总负债除以总资产]
        Δ LIQUIDITY(Change in Liquidity): 当前期流动比率减去上一期流动比率
    """
    
    def calculate(self):
        # ROA
        roa = ROA(self.prefix).calculate()
        indi_1 = roa.copy()
        indi_1[indi_1>0] = 1
        indi_1[indi_1<=0] = 0
        
        # delta_roa 
        delta_roa = ROA(self.prefix).calculate() - ROA(self.prefix).calculate(is_ttm=False,is_avg=True).shift(252)
        indi_2 = delta_roa.copy()
        indi_2[indi_2>0] = 1
        indi_2[indi_2<=0] = 0
        
        # CFO
        cfo = self.dl.get_financial_field_df('mananetr') 
        indi_3 = cfo.copy()
        indi_3[indi_3>0]=1
        indi_3[indi_3<=0]=0
        
        # ACCURALS
        netprofit = self.dl.get_financial_field_df('parenetp')
        accurals = netprofit - cfo
        indi_4 = accurals.copy()
        indi_4[indi_4<0]=1 # ACCRUAL小于0得1分,否则得0分。
        indi_4[indi_4>=0] = 0
        
        # LEVER
        delta_lever = DebtToAssetRate(self.prefix).calculate() - DebtToAssetRate(self.prefix).calculate().shift(252)
        indi_5 = delta_lever.copy()
        indi_5[indi_5<0]=1
        indi_5[indi_5>=0]=0
        
        # LiquidLever
        delta_liquidlever = LiquidityDebtToAssetRate(self.prefix).calculate() - LiquidityDebtToAssetRate(self.prefix).calculate().shift(252)
        indi_6 = delta_liquidlever.copy()
        indi_6[indi_6>=0]=1
        indi_6[indi_6<0]=0
        
        
        # Margin
        delta_margin = MarginRate(self.prefix).calculate() - MarginRate(self.prefix).calculate().shift(252)
        indi_7 = delta_margin.copy()
        indi_7[indi_7>0]=1
        indi_7[indi_7<=0]=0
        
        # Turnover
        delta_turnover = TurnoverRate(self.prefix).calculate() - TurnoverRate(self.prefix).calculate().shift(252)
        indi_8 = delta_turnover.copy()
        indi_8[indi_8>0]=1
        indi_8[indi_8<=0] = 1
        
        # Fscore
        fscore = indi_1 + indi_2 + indi_3 + indi_4 + indi_5 + indi_6 + indi_7 + indi_8
        
        return fscore