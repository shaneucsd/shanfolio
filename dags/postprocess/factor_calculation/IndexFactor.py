import numpy as np
import pandas as pd
import talib
from scipy import stats

from enums.index_estclass import IndexEstclass
from postprocess.datakit.DtLoader import DtLoader
from postprocess.datakit.DtTransform import DtTransform
from postprocess.factor_calculation.Factor import Factor
from utils.backtest_utils import get_benchmark_close
from utils.calculation_utils import calculate_listed_days, get_amount_over_limit

BM = "000985"
trans = DtTransform("index")
transforms = [trans.clean]


# 创建子类来处理不同的指数因子
class IndexClose(Factor):
    """
    Close

    Args:
        Factor (_type_): index
    """

    def calculate(self):
        close = self.dl.close.replace({0: np.nan}).ffill()
        return close


class IndexOpen(Factor):
    """
    Open
    Args:
        Factor (_type_): index
    """

    def calculate(self):
        open = self.dl.open.replace({0: np.nan}).ffill()
        return open


class IndexHigh(Factor):
    """
    High

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        high = self.dl.high.replace({0: np.nan}).ffill()
        return high


class IndexLow(Factor):
    """
    Low

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        low = self.dl.low.replace({0: np.nan}).ffill()
        return low


class IndexReturns(Factor):
    """
    Returns

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        returns = close.pct_change() * 100
        return returns


class IndexPreReturns(Factor):
    """
    PreReturns to benchmark

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        bm_close = get_benchmark_close(BM)
        returns = close.pct_change() * 100
        bm_returns = bm_close.pct_change() * 100
        pre_returns = returns.sub(bm_returns, axis=0)
        return pre_returns


class IndexWeekPchg(Factor):
    """
    Weekly Returns

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        week_pchg = close.pct_change(5) * 100
        return week_pchg


class IndexPreWeekPchg(Factor):
    """
    Weekly Pre Returns to benchmark

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        bm_close = get_benchmark_close(BM)

        week_pchg = close.pct_change(5) * 100
        bm_week_pchg = bm_close.pct_change(5) * 100
        pre_week_pchg = week_pchg.sub(bm_week_pchg, axis=0)

        return pre_week_pchg


class IndexMonthPchg(Factor):
    """
    Monthly Returns to benchmark

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        month_pchg = close.pct_change(20) * 100
        return month_pchg


class IndexPreMonthPchg(Factor):
    """
    Pre Monthly Returns to benchmark

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        close = IndexClose(self.prefix, self.index_estclass).calculate()
        bm_close = get_benchmark_close(BM)

        month_pchg = close.pct_change(20) * 100
        bm_month_pchg = bm_close.pct_change(20) * 100
        pre_month_pchg = month_pchg.sub(bm_month_pchg, axis=0)
        return pre_month_pchg


class IndexVolume(Factor):
    """
    Volume

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        volume = self.dl.volume.replace({0: np.nan}).ffill()
        return volume


class IndexAmount(Factor):
    """
    Amount

    Args:
        Factor (_type_): index

    """

    def calculate(self):
        amount = self.dl.amount.replace({0: np.nan}).ffill()
        return amount


class IndexVolumeQuantile(Factor):
    """
    Time series quantile of volume

    Args:
        Factor (_type_): index
        time_period (int): time period

    """

    def calculate(self, time_period=3 * 252):
        volume = self.dl.volume.replace({0: np.nan}).ffill()
        volume_quantile = volume.rolling(time_period).rank(pct=True)
        return volume_quantile


class IndexAmountQuantile(Factor):
    """
    Time series quantile of amount

    Args:
        Factor (_type_): index
        time_period (int): time period

    """

    def calculate(self, time_period=3 * 252):
        amount = self.dl.amount.replace({0: np.nan}).ffill()
        amount_quantile = amount.rolling(time_period).rank(pct=True)
        return amount_quantile


class IndexNegotiableMv(Factor):
    """
    NegotiableMv

    Args:
        Factor (_type_): index
        time_period (int): time period

    """

    def calculate(self):
        negotiablemv = self.dl.negotiablemv.replace({0: np.nan}).ffill()
        return negotiablemv


class IndexMADiffusionRatio(Factor):
    """
    N日均线以上的比例

    Args:
        Factor (_type_): index, estclass
        field (str): field name
        time_period (int): time period of criteria of ma
        time_period_ma (int): time period of moving average of diffusion ratio

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field=None, time_period=None, time_period_ma=None):
        strong_ratio = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )
        # N日均线以上的比例
        field = self.stock_data_loader.__getattr__("close")
        field_ma = field.gt(field.rolling(time_period).mean())  # N日均线以上的比例

        for index, components in self.dl.index_comp_dict.items():
            if index not in self.dl.tradeassets:
                continue
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.close.loc[:, components]
            field_ma_of_a_index = field_ma.loc[:, components]
            # Gt than Ratio / Total Asset count
            strong_ratio[index] = field_ma_of_a_index.sum(
                axis=1
            ) / field_of_a_index.notna().sum(axis=1)

        if time_period_ma is not None:
            strong_ratio = strong_ratio.rolling(time_period_ma).mean()

        return strong_ratio


class IndexVolumeRatio(Factor):
    """
    截面成交量占比

    Args:
        Factor (_type_): index, estclass

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self):
        volume = IndexVolume(self.prefix, self.index_estclass).calculate()
        volume_ratio = volume.div(volume.sum(axis=1), axis=0)
        return volume_ratio


class IndexAmountRatio(Factor):
    """
    截面成交额占比

    Args:
        Factor (_type_): index, estclass

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self):
        amount = IndexAmount(self.prefix, self.index_estclass).calculate()
        amount_ratio = amount.div(amount.sum(axis=1), axis=0)
        return amount_ratio


class IndexVolumeRatioHistQuantile(Factor):
    """
    截面成交量占比: 历史分位数

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    """

    def calculate(self, time_period=252 * 3):
        volume_ratio = IndexVolumeRatio(self.prefix, self.index_estclass).calculate()
        volume_ratio_quantile = volume_ratio.rolling(time_period).rank(pct=True)
        return volume_ratio_quantile


class IndexAmountRatioHistQuantile(Factor):
    """
    截面成交额占比: 历史分位数

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    """

    def calculate(self, time_period=252 * 3):
        amount_ratio = IndexAmountRatio(self.prefix, self.index_estclass).calculate()
        amount_ratio_quantile = amount_ratio.rolling(time_period).rank(pct=True)
        return amount_ratio_quantile


class IndexExtremeUpCount(Factor):
    """
    涨停家数

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self):
        # 涨停家数
        extreme_up_count = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )
        pchg = self.stock_data_loader.close.replace({0: np.nan}).ffill().pct_change()
        pchg_is_extreme_up = pchg.gt(0.099)  # 涨停家数

        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            extreme_up_count[index] = pchg_is_extreme_up.loc[:, components].sum(axis=1)
        return extreme_up_count


class IndexExtremeDownCount(Factor):
    """
    跌停家数

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self):
        # 涨停家数
        extreme_down_count = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )
        pchg = self.stock_data_loader.close.replace({0: np.nan}).ffill().pct_change()
        pchg_is_extreme_down = pchg.lt(-0.099)

        for index, components in self.dl.index_comp_dict.items():
            if index not in self.dl.tradeassets:
                continue
            components = self.stock_data_loader.close.columns.isin(components)
            extreme_down_count[index] = pchg_is_extreme_down.loc[:, components].sum(
                axis=1
            )
        return extreme_down_count


class IndexHinderburgDelta(Factor):
    """
    兴登堡预兆
    净新高占比：（新高家数-新低家数）/（总家数-新上市家数）
    标准化：（净新高占比-净新高占比N日均值）/净新高占比N日标准差

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period
        time_period_ma (int): time period of moving average of hinderburg delta

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, time_period=252, time_period_ma=3 * 252):
        hinderburg_delta = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )

        close = self.stock_data_loader.close.replace({0: np.nan}).ffill()
        listed_days = close.apply(calculate_listed_days, axis=0)
        new_low_df = (
            close.rolling(time_period).min().eq(close) & (listed_days > 250)
        ).astype(int)
        new_high_df = close.rolling(time_period).max().eq(close) & (
            listed_days > 250
        ).astype(int)

        for index, components in self.dl.index_comp_dict.items():
            components = close.columns.isin(components)

            new_high_df_of_a_index = new_high_df.loc[:, components]
            new_low_df_of_a_index = new_low_df.loc[:, components]
            listed_days_of_a_index = listed_days.loc[:, components]

            count_of_stocks_wo_newly_listed = listed_days_of_a_index[
                listed_days_of_a_index > 250
            ].sum(axis=1)

            hinderburg_delta[index] = (
                new_high_df_of_a_index.sum(axis=1) - new_low_df_of_a_index.sum(axis=1)
            ) / count_of_stocks_wo_newly_listed

        hinderburg_delta = (
            hinderburg_delta - hinderburg_delta.rolling(time_period_ma).mean()
        ) / (hinderburg_delta.rolling(time_period_ma).std())

        return hinderburg_delta


class IndexHinderburgDeltaOfTopTraded(Factor):
    """
    兴登堡预兆: 成交额过低剔除
    净新高占比：（新高家数-新低家数）/（总家数-新上市家数）
    标准化：（净新高占比-净新高占比N日均值）/净新高占比N日标准差
    剔除成交额过低的股票后再计算

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period
        time_period_ma (int): time period of moving average of hinderburg delta

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, time_period=252, time_period_ma=252 * 3):
        hinderburg_delta = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )

        close = self.stock_data_loader.close.replace({0: np.nan}).ffill()
        amount = self.stock_data_loader.amount.replace({0: np.nan}).ffill()

        listed_days = close.apply(calculate_listed_days, axis=0)

        new_low_df = (
            close.rolling(time_period).min().eq(close) & (listed_days > 250)
        ).astype(int)
        new_high_df = close.rolling(time_period).max().eq(close) & (
            listed_days > 250
        ).astype(int)

        for index, components in self.dl.index_comp_dict.items():
            components = close.columns.isin(components)
            amount_of_a_index = amount.loc[:, components]
            selected_stocks = get_amount_over_limit(amount_of_a_index)

            new_high_df_of_a_index = new_high_df.loc[:, components] * selected_stocks
            new_low_df_of_a_index = new_low_df.loc[:, components] * selected_stocks
            listed_days_of_a_index = listed_days.loc[:, components] * selected_stocks

            count_of_stocks_wo_newly_listed = listed_days_of_a_index[
                listed_days_of_a_index > 250
            ].sum(axis=1)

            hinderburg_delta[index] = (
                new_high_df_of_a_index.sum(axis=1) - new_low_df_of_a_index.sum(axis=1)
            ) / count_of_stocks_wo_newly_listed

        hinderburg_delta = (
            hinderburg_delta - hinderburg_delta.rolling(time_period_ma).mean()
        ) / (hinderburg_delta.rolling(time_period_ma).std())

        return hinderburg_delta


# Returns
class IndexLogReturn(Factor):
    """
    对数收益率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    """

    def calculate(self, time_period=None):
        return np.log(self.dl.close) - np.log(self.dl.close).shift(time_period)


class IndexCumReturn(Factor):
    """
    累计m日收益率(n日涨跌幅)

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period
        time_period2 (int): time period of rolling sum

    Default:
        time_period = 20

    """

    def calculate(self, time_period=1, time_period2=252):
        bm_close = get_benchmark_close(BM)
        ret = self.dl.close.pct_change(time_period)
        bm_ret = bm_close.pct_change(time_period)
        pre_ret = ret.sub(bm_ret, axis=0)
        cum_ret = pre_ret.rolling(time_period2).sum()
        return cum_ret


class IndexGoldenReturn(Factor):
    """
    黄金律收益率

    # 日内收益率排名 - 隔日收益率排名

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period

    Default:
        time_period = 20

    """

    def calculate(self, time_period=20):
        intra_ret = self.dl.close.divide(self.dl.open, axis=0) - 1
        intra_ret = intra_ret.rolling(time_period).sum()

        post_ret = self.dl.open.divide(self.dl.close.shift(1), axis=0) - 1
        post_ret = post_ret.rolling(time_period).sum()

        golden_ret = intra_ret.rank(axis=1) - post_ret.rank(axis=1)

        return golden_ret


class IndexMomentumSpread(Factor):
    """
    期限内动量差 (过去n日收益率 - 过去m日收益率)

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term
        time_period2 (int): time period of Short-term

    Default:
        time_period = 10
        time_period2 = 5

    """

    def calculate(self, time_period=10, time_period2=5):
        if time_period >= time_period2:
            ls_mom_spread = IndexLogReturn(self.prefix, self.index_estclass).calculate(
                time_period=time_period
            ) - IndexLogReturn(self.prefix, self.index_estclass).calculate(
                time_period=time_period2
            )
        else:
            raise KeyError(
                f"IndexMomentumSpread 要求 time_period ({time_period}) > time_period2 ({time_period2}) "
            )
        return ls_mom_spread


class IndexRSI(Factor):
    """
    RSI

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    Default:
        time_period = 120

    """

    def calculate(self, time_period=120):
        rsi = self.dl.close.apply(lambda x: talib.RSI(x, time_period))
        return rsi


class IndexReturnAmountAdjusted(Factor):
    """
    成交额调整动量

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def calculate(self, time_period=252):
        ret = IndexPreReturns(self.prefix, self.index_estclass).calculate()
        ret_amt_adj = ret / self.dl.amount
        ret_amt_adj = (ret_amt_adj.rolling(time_period).mean()) / ret_amt_adj.rolling(
            time_period
        ).std()
        return ret_amt_adj


class IndexBIAS(Factor):
    """
    乖离率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def calculate(self, time_period=20):
        bias = self.dl.close / self.dl.close.rolling(time_period).mean() - 1
        return bias


#  Volatility
class IndexDailyStd(Factor):
    """
    乖离率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def calculate(self, time_period=252):
        # log_returns = np.log(self.dl.close) - np.log(self.dl.close).shift(1)
        print("x")
        pre_returns = IndexPreReturns(self.prefix, self.index_estclass).calculate()
        dastd = pre_returns.rolling(time_period).std()
        return dastd


class IndexCrossVolumeStd(Factor):
    """
    截面成交量波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="volume"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        return field_std


class IndexCrossAmountStd(Factor):
    """
    截面成交额波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="amount"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        return field_std


class IndexCrossTurnrateStd(Factor):
    """
    截面换手率波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="turnrate"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        return field_std


class IndexRankCrossVolumeStd(Factor):
    """
    截面成交量波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="volume", time_period=252 * 3):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std_rank


class IndexCrossVolatilityStd(Factor):
    """
    截面成交量波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="close"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        close = self.stock_data_loader.__getattr__(field).ffill()
        volatility = close.pct_change().rolling(20).std()
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = volatility.loc[:, components]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        # field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std


class IndexCrossVolatilityStd2(Factor):
    """
    截面成交量波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="close"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        close = self.stock_data_loader.__getattr__(field).ffill()
        volatility = close.pct_change().rolling(20).std()
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = volatility.loc[:, components]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).std()
        # field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std


class IndexRankCrossVolatilityStd(Factor):
    """
    截面成交量波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="close", time_period=252 * 3):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        close = self.stock_data_loader.__getattr__(field).ffill()
        volatility = close.pct_change().rolling(20).std()
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = volatility.loc[:, components]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std_rank


class IndexRankCrossAmountStd(Factor):
    """
    截面成交额波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="amount", time_period=252 * 3):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std_rank


class IndexRankCrossTurnrateStd(Factor):
    """
    截面换手率波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="turnrate", time_period=252 * 3):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std_rank


class IndexStdCrossTurnrateStd(Factor):
    """
    截面换手率波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="turnrate"):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        return field_std


class IndexRankStdCrossTurnrateStd(Factor):
    """
    截面换手率波动率,的时序波动率

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field="turnrate", time_period=252 * 3):
        field_std = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)
        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            field_of_a_index = self.stock_data_loader.__getattr__(field).loc[
                :, components
            ]
            field_std[index] = field_of_a_index.std(axis=1)
        field_std = field_std.rolling(20).mean()
        field_std_rank = field_std.rolling(time_period).rank(pct=True)
        return field_std_rank


# Forecast
class IndexFcst(Factor):
    #     'eps_fcst_mv_yoy2': ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='yoy2'),
    #     'eps_fcst_mv_yoy':ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='yoy'),
    #     'eps_fcst_mv_quantile':ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='quantile'),
    #     'roe_fcst_mv_yoy': ICF_fcst.calculate(method='mv',name='roe_fcst',diff_method='yoy'),
    #     'roe_fcst_mv_quantile':ICF_fcst.calculate(method='mv',name='roe_fcst',diff_method='quantile'),
    # Common Parameters:
    #  eps_fcst, totmktcap,
    #  eps_fcst, totmktcap,
    #  roe_fcst, totmktcap
    """
    成分股一致预期数据

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, field=None, weight_type=None, comparison_type=None):
        """
        :param field:
        :param weight_type:
            {
            'totmktcap','negotiablemv','median','mean'
            }
        :return:
        """
        fcst = self.stock_data_loader.__getattr__(field).replace({0: np.nan}).ffill()
        index_fcst = pd.DataFrame(index=self.dl.tradedays, columns=self.dl.tradeassets)

        for index, components in self.dl.index_comp_dict.items():
            components = self.stock_data_loader.close.columns.isin(components)
            fcst_of_a_index = fcst.loc[:, components]

            # 根据个股加权
            if weight_type == "totmktcap":
                totmktcap_of_a_index = self.stock_data_loader.totmktcap.loc[
                    :, components
                ]
                stock_weight = totmktcap_of_a_index.div(
                    totmktcap_of_a_index.sum(axis=1), axis=0
                )
                fcst_of_a_index = fcst_of_a_index.mul(stock_weight, axis=0).sum(axis=1)

            elif weight_type == "negotiablemv":
                negotiablemv_of_a_index = self.stock_data_loader.negotiablemv.loc[
                    :, components
                ]
                stock_weight = negotiablemv_of_a_index.div(
                    negotiablemv_of_a_index.sum(axis=1), axis=0
                )
                fcst_of_a_index = fcst_of_a_index.mul(stock_weight, axis=0).sum(axis=1)

            elif weight_type == "median":
                fcst_of_a_index = fcst_of_a_index.median(axis=1)

            elif weight_type == "mean":
                fcst_of_a_index = fcst_of_a_index.mean(axis=1)

            else:
                raise KeyError(f"不支持的加权方式, {weight_type}")

            # 对Fcst结果做变化
            if comparison_type == "raw":
                fcst_of_a_index = fcst_of_a_index

            elif comparison_type == "mom":
                fcst_of_a_index = fcst_of_a_index.pct_change(20)

            elif comparison_type == "mom2":
                fcst_of_a_index = fcst_of_a_index.pct_change(20).diff(20)

            elif comparison_type == "yoy":
                fcst_of_a_index = fcst_of_a_index.pct_change(252)

            elif comparison_type == "yoy2":
                fcst_of_a_index = fcst_of_a_index.pct_change(252).diff(252)

            elif comparison_type == "quantile":
                fcst_of_a_index = fcst_of_a_index.rolling(3 * 250).rank(pct=True)
            else:
                raise KeyError(f"不支持的加权方式, {comparison_type}")

            index_fcst[index] = fcst_of_a_index

        index_fcst = trans.run(
            index_fcst.copy(),
            transforms=transforms,
        )

        return index_fcst


class IndexAbsoluteNorthMoney(Factor):
    """
    北向资金净流入占比

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    # na_fs = {
    #     'na_active_raw': IndexAbsoluteNorthMoney('index', IndexEstclass.SWLEVEL1_INDEX).calculate(time_period=120,
    #                                                                                               components_pool='industry',
    #                                                                                               weight_type='amt',
    #                                                                                               amount_count_type='net'),
    #     'na_active_abs': IndexAbsoluteNorthMoney('index', IndexEstclass.SWLEVEL1_INDEX).calculate(time_period=120,
    #                                                                                               components_pool='industry',
    #                                                                                               weight_type='amt',
    #                                                                                               amount_count_type='abs'),
    #     'na_mktcap_ratio': IndexAbsoluteNorthMoney('index', IndexEstclass.SWLEVEL1_INDEX).calculate(time_period=120,
    #                                                                                                 components_pool='industry',
    #                                                                                                 weight_type='totmktcap',
    #                                                                                                 amount_count_type='net'),
    # }
    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(
        self,
        time_period=None,
        weight_type=None,
        components_pool=None,
        amount_count_type=None,
    ):
        north_amt_net_inflow_ratio = pd.DataFrame(
            index=self.dl.tradedays, columns=self.dl.tradeassets
        )
        # 北向成交股数
        if amount_count_type == "absolute":
            abs_north_share_diff = self.stock_data_loader.shhksharehold.diff().abs()
        elif amount_count_type == "net":
            abs_north_share_diff = self.stock_data_loader.shhksharehold.diff()
        else:
            raise KeyError(f"不支持的参数 amount_count_type : {amount_count_type}")

        # 北向成交金额 = 收盘价 * 北向成交股数
        north_money_amount = self.stock_data_loader.close.mul(
            abs_north_share_diff, axis=0
        ).fillna(0)

        # 北向参与的股票
        stocks_north_money_buy = abs_north_share_diff.columns.intersection(
            self.stock_data_loader.vwap.columns
        )

        for index, components in self.dl.index_comp_dict.items():
            # 仅计算北向成交的股票
            if components_pool == "north_only":
                components = [i for i in stocks_north_money_buy if i in components]
            # 仅计算行业成分股
            elif components_pool == "industry":
                components = [
                    i for i in self.stock_data_loader.close.columns if i in components
                ]
            else:
                raise KeyError(f"不支持的参数 components_pool : {components_pool}")

            # 北向n日净流入金额
            north_money_amount_of_a_index = north_money_amount.loc[:, components]
            north_money_amount_of_a_index = north_money_amount_of_a_index.rolling(
                time_period
            ).sum()
            # 成分股n日总成交金额
            amount_of_a_index = self.stock_data_loader.amount.loc[:, components]
            amount_of_a_index = amount_of_a_index.rolling(time_period).sum()

            ratio = north_money_amount_of_a_index / amount_of_a_index

            if weight_type == "totmktcap":
                # 成分股n日总加权的北向成交占比
                totmktcap = self.stock_data_loader.totmktcap.loc[:, components]
                totmktcap = totmktcap.div(totmktcap.sum(axis=1), axis=0)
                north_amt_ratio = ratio.mul(totmktcap, axis=0).sum(axis=1)

            elif weight_type == "negotiablemv":
                # 成分股n日流通市值加权的北向成交占比
                negotiablemv = self.stock_data_loader.negotiablemv.loc[:, components]
                negotiablemv = negotiablemv.div(negotiablemv.sum(axis=1), axis=0)
                north_amt_ratio = ratio.mul(negotiablemv, axis=0).sum(axis=1)
            elif weight_type == "amt":
                # 成分股n日成交金额加权的北向成交占比
                amt = self.stock_data_loader.amount.loc[:, components]
                amt = amt.div(amt.sum(axis=1), axis=0)
                north_amt_ratio = ratio.mul(amt, axis=0).sum(axis=1)

            elif weight_type == "raw":
                north_amt_ratio = north_money_amount_of_a_index.sum(
                    axis=1
                ) / amount_of_a_index.sum(axis=1)
            else:
                raise KeyError(f"不支持的加权方式, {weight_type}")

            north_amt_net_inflow_ratio[index] = north_amt_ratio

        return north_amt_net_inflow_ratio


class IndexFundFlowRatio(Factor):
    # IndexFundFlowRatio("index",IndexEstclass.SWLEVEL1_INDEX).calculate(time_period=20,inflow=f'sinflow',outflow=f'soutflow',count_type=1)
    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(
        self,
        time_period=20,
        inflow="sinflow",
        outflow="soutflow",
        count_type=1,
        weight_type="raw",
    ):
        nf_ratio = pd.DataFrame(index=self.dl.close.index, columns=self.dl.tradeassets)
        if type(inflow) == list:
            inflow, outflow = self.stock_data_loader.__getattr__(
                inflow[0]
            )+self.stock_data_loader.__getattr__(
                inflow[1]
            ), self.stock_data_loader.__getattr__(outflow[0])+ self.stock_data_loader.__getattr__(outflow[1])
                
        else:
            inflow, outflow = self.stock_data_loader.__getattr__(
                inflow
            ), self.stock_data_loader.__getattr__(outflow)
            
        factor = inflow - outflow
        if count_type == 1:
            abs_netflow = (inflow - outflow).abs()
        elif count_type == 2:
            abs_netflow = self.dl.amount
        elif count_type == 3:
            abs_netflow = inflow + outflow
        elif count_type == 4:
            abs_netflow = inflow.abs() + outflow.abs()
        else:
            raise KeyError(f"不支持的参数 count_type : {count_type}")

        for index, components in self.dl.index_comp_dict.items():
            group = [i for i in components if i in factor.columns]
            nf = factor.loc[:, group]
            nf = nf.rolling(time_period).sum().sum(axis=1)

            if weight_type == "totmkt":
                nmv = self.stock_data_loader.totmktcap.loc[:, group].sum(axis=1)
                nf_ratio[index] = nf.div(nmv, axis=0)
            elif weight_type == "nmv":
                nmv = self.stock_data_loader.negotiablemv.loc[:, group].sum(axis=1)
                nf_ratio[index] = nf.div(nmv, axis=0)
            elif weight_type == "amount":
                nmv = self.stock_data_loader.amount.loc[:, group].sum(axis=1)
                nf_ratio[index] = nf.div(nmv, axis=0)
            elif weight_type == "raw":
                abs_f = abs_netflow.loc[:, group]
                abs_f = abs_f.rolling(time_period).sum().sum(axis=1)
                nf_ratio[index] = nf.div(abs_f, axis=0)
            else:
                raise KeyError(f"不支持的参数 weight_type : {weight_type}")
        return nf_ratio


class IndexSmallFundCorrelation(Factor):
    """
    小单资金流

    Args:
        Factor (_type_): index, estclass
        time_period (int): time period of Long-term

    """

    # Factor:
    # N = 20
    # 小单资金净流入 与 前日收益率的相关系数
    # IndexSmallFundCorrelation('index',IndexEstclass.SWLEVEL1_INDEX).calculate(time_period = 20)

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, time_period=20):
        s_corr = pd.DataFrame(index=self.dl.close.index, columns=self.dl.tradeassets)
        s_net = self.stock_data_loader.sinflow - self.stock_data_loader.soutflow
        ret = self.dl.close.ffill().pct_change()

        delta_s_net = s_net.diff()
        delta_ret = ret.diff().shift(1)

        for index, components in self.dl.index_comp_dict.items():
            components = [i for i in components if i in delta_s_net.columns]
            components = list(set(components))
            idx = delta_s_net.index.intersection(delta_ret.index)

            delta_s_net_of_an_index = delta_s_net.loc[idx, components]
            delta_s_net_of_an_index = delta_s_net_of_an_index.sum(axis=1)

            delta_ret_of_an_index = delta_ret.loc[idx, index]

            _corr = delta_s_net_of_an_index.rolling(time_period).apply(
                lambda s: stats.spearmanr(s, delta_ret_of_an_index.loc[s.index])[0]
            )

            s_corr[index] = _corr

        return s_corr


class IndexChipDistributionReturn(Factor):
    # 筹码收益率
    # Factor
    # IndexChipDistributionReturn('index',IndexEstclass.SWLEVEL1_INDEX).calculate(time_period = 15)
    # IndexChipDistributionReturn('index',IndexEstclass.SWLEVEL1_INDEX).calculate(time_period = 252)

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, time_period=None):
        chip = pd.DataFrame(
            index=self.dl.close.index,
            columns=self.dl.tradeassets,
        )
        close = self.stock_data_loader.close
        turnrate = self.stock_data_loader.turnrate
        rc = close.pct_change(time_period)

        for index, components in self.dl.index_comp_dict.items():
            components = [
                i for i in components if i in close.columns and i in turnrate.columns
            ]

            rc_comp = rc.loc[:, components]
            turnrate_comp = turnrate.loc[:, components]

            atr = turnrate_comp.shift(time_period)

            for n in range(0, time_period):
                atr *= 1 - turnrate_comp.shift(n)

            arc = (
                atr.rolling(time_period)
                .sum()
                .mul(rc_comp, axis=0)
                .div(atr.rolling(time_period).sum(), axis=0)
            )

            chip[index] = arc.mean(axis=1)

        return chip


class IndexStrongStockRatio(Factor):
    # strong_ratio_gt220_3 =  ICF_strong_ratio('index',IndexEstclass.SWLEVEL1_INDEX).calculate(fit='gt220',method=1)

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(self, fit=None, method=None):
        strong_ratio = pd.DataFrame(
            index=self.dl.close.index, columns=self.dl.tradeassets
        )
        close = self.stock_data_loader.close

        for index, components in self.dl.index_comp_dict.items():
            components = [
                i
                for i in components
                if i in self.stock_data_loader.close.columns.tolist()
            ]
            close_of_an_index = close.loc[:, components]

            if fit == "ma60":
                stock_is_fit = (
                    (close_of_an_index.rolling(60).mean().notna())
                    & (close_of_an_index.gt(close_of_an_index.rolling(60).mean()))
                ).astype(int)
            elif fit == "ma120":
                stock_is_fit = (
                    (close_of_an_index.rolling(120).mean().notna())
                    & (close_of_an_index.gt(close_of_an_index.rolling(120).mean()))
                ).astype(int)
            elif fit == "gt220":
                stock_is_fit = (
                    (close_of_an_index.shift(220).notna())
                    & (close_of_an_index.gt(close_of_an_index.shift(220)))
                ).astype(int)
            else:
                raise KeyError(f"不支持的参数 fit : {fit}")

            if method == 1:
                strong_ratio[index] = stock_is_fit.mean(axis=1)
            elif method == 2:
                nmv = self.stock_data_loader.negotiablemv.loc[:, components]
                nmv = nmv.div(nmv.sum(axis=1), axis=0)
                strong_ratio[index] = (stock_is_fit.multiply(nmv, axis=0)).sum(axis=1)
            elif method == 3:
                mv = self.stock_data_loader.totmktcap.loc[:, components]
                mv = mv.div(mv.sum(axis=1), axis=0)
                strong_ratio[index] = (stock_is_fit.multiply(mv, axis=0)).sum(axis=1)
            elif method == 4:
                amt = self.stock_data_loader.amount.loc[:, components]
                amt = amt.div(amt.sum(axis=1), axis=0)
                strong_ratio[index] = (stock_is_fit.multiply(amt, axis=0)).sum(axis=1)
            else:
                raise KeyError(f"不支持的参数 method : {method}")

        return strong_ratio


class IndexStrongStockRatioShortToLong(Factor):
    # strong_ratio_gt220_4_2_1_60 = IndexStrongStockRatioShortToLong('index',IndexEstclass.SWLEVEL1_INDEX).calculate(fit='gt220',method=4,type='2',time_period1=1,time_period2=60)

    def __init__(self, prefix, index_estclass: IndexEstclass = None):
        super().__init__(prefix, index_estclass)
        self.stock_data_loader = DtLoader(prefix="stock")

    def calculate(
        self, fit=None, method=None, type=None, time_period1=None, time_period2=None
    ):
        strong_ratio = IndexStrongStockRatio(
            self.prefix, self.index_estclass
        ).calculate(fit=fit, method=method)
        if type == "1":
            strong_ratio = strong_ratio.diff(time_period1)
        if type == "2":
            strong_ratio = (
                strong_ratio.rolling(time_period1).mean()
                / strong_ratio.rolling(time_period2).mean()
            )
        if type == "3":
            strong_ratio = (
                strong_ratio.rolling(time_period1).mean()
                / strong_ratio.rolling(time_period2).mean()
            )
            strong_ratio = strong_ratio.diff(time_period2)
        return strong_ratio


# momentum_fs = {
#     'cum_ret':cum_ret,               11111111111111111111111111111111111
#     'ls_mom_spread_10_5':ls_mom_spread_10_5,    11111111111111111111111111111111111
#     'rsi':F_RSI.calculate(window=120),   11111111111111111111111111111111111
#     'golden_ret':golden_ret,   11111111111111111111111111111111111
#     'ret_amt_adj':ret_amt_adj,   11111111111111111111111111111111111
# }
#
# volatility_fs ={'volume_std':volume_std, 'amount_std':amount_std} 11111111111111111111111111111111111
#
#
# fcst= {
#     'eps_fcst_mv_yoy2': ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='yoy2'),   11111111111111111111111111111111111
#     'eps_fcst_mv_yoy':ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='yoy'),   11111111111111111111111111111111111
#     'eps_fcst_mv_quantile':ICF_fcst.calculate(method='mv',name='eps_fcst',diff_method='quantile'),   11111111111111111111111111111111111
#     'roe_fcst_mv_yoy': ICF_fcst.calculate(method='mv',name='roe_fcst',diff_method='yoy'),   11111111111111111111111111111111111
#     'roe_fcst_mv_quantile':ICF_fcst.calculate(method='mv',name='roe_fcst',diff_method='quantile'),   11111111111111111111111111111111111
# }
#
# 1111111
# na_fs = {
#     'na_active_raw':ICF_north_amt_net_inflow_ratio.calculate(window=120,group='industry',method='amt',diff='raw'),
#     'na_active_abs':ICF_north_amt_net_inflow_ratio.calculate(window=120,group='industry',method='amt',diff='abs'),
#     'na_mktcap_ratio':ICF_north_amt_net_inflow_ratio.calculate(window=120,group='industry',method='totmktcap',diff='raw'),
#  }
#
# mf_fs = \
# {
#     'mf_extra': ICF_mutual_fund_pos.calculate(window = 0, pos_type = 'extra', method = 'amt', mutual_fund_pos = mutual_fund_pos, bm = bm),
#     'mf_extra_yoy': ICF_mutual_fund_pos.calculate(window = 252, pos_type = 'extra', method = 'amt', mutual_fund_pos = mutual_fund_pos, bm = bm),
# }
#
# growth_fs = {
#     'roe_sue':ICF_sue.calculate(name='roe',method='sue',q=0.5).ffill(),
#     'roe_yoy':ICF_sue.calculate(name='roe',method='yoy',q=0.5).ffill(),
# }
#
# fund_flow_fs = {
#     's_strength':s_strength,
#     # 'etral_nmv':etral_totmkt,
#     's_corr':s_corr
# }
#
#
# dfu_fs = {
#     'dfu_s2l':strong_ratio_gt220_4_2_1_60,
#     'dfu':strong_ratio_gt220_3,
# }
#
# arc_fs = {
#     'arc_15':ICF_arc.calculate(N=15),
#     'arc_252':ICF_arc.calculate(N=252),
# }
