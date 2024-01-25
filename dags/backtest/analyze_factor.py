import pathlib
import sys
import empyrical
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from enums.index_estclass import IndexEstclass
from utils.file_utils import (
    PROCESSED_FACTOR_PATH,
    load_object_from_pickle,
    load_object_from_pickle_with_path,
)
from utils.backtest_utils import (
    get_benchmark_close,
    align_data_w_stock_pool,
    align_data_wo_stock_pool,
)
from postprocess.datakit.DtLoader import DtLoader
from postprocess.factor_calculation import StockFactor


class Analyzer:
    """
    _summary_
    : Read Local Factor data.

    """

    def __init__(
        self,
        prefix: str = "stock",
        index_estclass: IndexEstclass = None,
        freq: str = "M",
        n_quantiles: int = 5,
        start: str = "2012-01-01",
        end: str = "2022-12-31",
        benchmark_symbol: str = "000300",
        ascending: int = 1,
        scope: str = None,
        commission_rate: float = 0.00,
        slippage_rate: float = 0.00,
    ) -> None:
        """
        初始化分析器。

        :param prefix: 数据前缀。
        :param index_estclass: 索引估算类。
        :param freq: 数据频率。
        :param n_quantiles: 分位数数量。
        :param start: 开始日期。
        :param end: 结束日期。
        :param benchmark_symbol: 基准符号。
        :param ascending: 是否升序排序。
        :param scope: 分析范围。
        :param commission_rate: 佣金率。
        :param slippage_rate: 滑点率。
        """
        self.prefix = prefix
        self.index_estclass = index_estclass
        self.dl = DtLoader(prefix=prefix, index_estclass=index_estclass)

        self.start = start
        self.end = end
        self.freq = freq
        self.n_quantiles = n_quantiles
        self.benchmark_symbol = benchmark_symbol
        self.period = {"M": 12, "W": 52, "D": 252}[self.freq]
        self.ascending = ascending
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.set_target_scope(scope)
        self.target_dir = f"{PROCESSED_FACTOR_PATH}/{self.prefix}"

    def set_target_scope(self, scope):
        # 设置股票池
        if self.prefix == "stock" and scope is not None:
            self.scope = scope
            self.stock_pool = load_object_from_pickle(f"stock_pool_{self.scope}")
        else:
            self.scope = None
            self.stock_pool = None

    def set_target_factor(self, factor_name,second_dir=None, calc=True):
        # 设置
        self.factor_name = factor_name
        tmp_dir = self.target_dir if not second_dir else self.target_dir+ '/' + second_dir
        self.factor = load_object_from_pickle_with_path(factor_name, tmp_dir)
        self.factor = self.factor * self.ascending
        if calc:
            self.prepare_factor()
            self.get_factor_group_returns_with_fee(hard_calc=True)
            self.indicators = pd.DataFrame(self.get_indicators(), index=[0]).round(4).T
            
            

    def get_factor_name_list(self):
        # find all the factor names in the target_dir
        return [
            file.stem
            for file in pathlib.Path(self.target_dir).glob("*")
            if not file.stem.startswith(".")
        ]

    def prepare_factor(self):
        self.trade_price = self.dl.close.ffill()
        self.benchmark_price = get_benchmark_close(self.benchmark_symbol)
        (
            self.factor,
            self.trade_price,
            self.future_returns,
            self.benchmark_returns,
        ) = align_data_w_stock_pool(
            self.factor,
            self.trade_price,
            self.benchmark_price,
            self.stock_pool,
            self.freq,
        )
        # align by start and end date
        self.factor = self.factor.loc[self.start : self.end]
        self.future_returns = self.future_returns.loc[self.start : self.end]
        self.benchmark_returns = self.benchmark_returns.loc[self.start : self.end]
        self.quantiled_factor = self.get_quantiled_factor()

    def get_quantiled_factor(self):
        """
        截面分组
        """

        def quantile_grouping(matrix, q=5):
            """
            对矩阵的每一行进行分位数分组。

            参数:
                matrix (numpy.ndarray): 输入的矩阵。
                q (int): 分组数量。
                ascending (bool): 是否按升序分组。

            返回:
                numpy.ndarray: 分组标签矩阵。
            """
            # 初始化输出矩阵
            output_matrix = np.zeros(matrix.shape)

            # 获取矩阵的行数
            num_rows = matrix.shape[0]

            for i in range(num_rows):
                row = matrix[i, :]
                valid_values = row[~np.isnan(row)]
                quantiles = np.nanpercentile(valid_values, np.linspace(0, 100, q + 1))
                labels = np.digitize(valid_values, quantiles, right=False)
                labels[labels == q + 1] = q
                output_matrix[i, ~np.isnan(row)] = labels
                output_matrix[i, np.isnan(row)] = -1

            return output_matrix

        # 示例调用
        quantiled_factor = quantile_grouping(
            self.factor.values, self.n_quantiles
        )  # 或者 False，取决于您的需要

        quantiled_factor = pd.DataFrame(
            quantiled_factor,
            index=self.factor.index,
            columns=self.factor.columns,
        )

        quantiled_factor.where(quantiled_factor != -1, np.nan, inplace=True)
        quantiled_factor.where(
            quantiled_factor != self.n_quantiles + 1, np.nan, inplace=True
        )

        return quantiled_factor

    def get_factor_group_returns(self, hard_calc=False):
        """
        分组收益率
        """
        if hasattr(self, "factor_group_returns") and not hard_calc:
            return self.factor_group_returns

        self.factor.index.name = "tradedate"

        factor_group_returns = pd.DataFrame(
            index=pd.to_datetime(self.factor.index),
            columns=range(1, self.n_quantiles + 1),
        )

        for i in range(1, self.n_quantiles + 1):
            # reset to boolean
            factor_group = self.quantiled_factor[self.quantiled_factor == i] / i
            # Need to consider the commission fee
            factor_group_returns[i] = (factor_group * self.future_returns).mean(axis=1)

        factor_group_returns["passive"] = self.future_returns.mean(axis=1)

        self.factor_group_returns = (
            factor_group_returns - factor_group_returns["passive"].values
        )

        return factor_group_returns

    def get_factor_group_returns_with_fee(self, hard_calc=False):
        """
        分组收益率，考虑手续费
        """
        if hasattr(self, "factor_group_returns_with_fee") and not hard_calc:
            return self.factor_group_returns

        factor_group_returns = pd.DataFrame(
            index=self.quantiled_factor.index,
            columns=range(1, self.n_quantiles + 1),
        )

        last_quantile = pd.DataFrame(columns=self.quantiled_factor.columns)

        for date in self.quantiled_factor.index:
            current_quantile = self.quantiled_factor.loc[date]
            if not last_quantile.empty:
                changing_stocks = current_quantile[current_quantile != last_quantile]
            else:
                changing_stocks = pd.Series(dtype=float)

            # 根据换仓比例计算手续费
            turnover_rate = changing_stocks.count() / current_quantile.count()
            daily_fee = turnover_rate * self.commission_rate

            for i in range(1, self.n_quantiles + 1):
                factor_group = current_quantile[current_quantile == i] / i
                factor_group_returns.loc[date, i] = (
                    factor_group * (self.future_returns.loc[date] - daily_fee)
                ).mean()

            last_quantile = current_quantile

        factor_group_returns = factor_group_returns.astype(float)
        factor_group_returns["passive"] = self.future_returns.mean(axis=1)

        self.factor_group_returns = factor_group_returns

        return factor_group_returns

    def get_factor_long_returns(self, is_log=False):
        """
        多头收益时序
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[self.n_quantiles]
        returns = np.log(1 + returns) if is_log else returns
        return returns

    def get_factor_short_returns(self, is_log=False):
        """
        多头收益时序
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[1]
        returns = np.log(1 + returns) if is_log else returns
        return returns

    def get_factor_long_short_returns(self, is_log=False):
        """
        多头收益时序
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[self.n_quantiles] - factor_group_returns[1]
        returns = np.log(1 + returns) if is_log else returns
        return returns

    def get_benchmark_returns(self, is_log=False):
        """
        多头收益时序
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns["passive"]
        returns = np.log(1 + returns) if is_log else returns
        return returns

    def get_factor_group_net_value(self, is_log=True, demeaned=True):
        """
        分组净值曲线
        """
        if demeaned:
            factor_group_returns = self.get_factor_group_returns().sub(
                self.get_factor_group_returns()["passive"], axis=0
            )
            factor_group_returns = factor_group_returns.drop(columns="passive")
        else:
            factor_group_returns = self.get_factor_group_returns()
        factor_group_net_value = (factor_group_returns + 1).cumprod()
        factor_group_net_value = (
            np.log(factor_group_net_value) + 1 if is_log else factor_group_net_value
        )
        return factor_group_net_value

    def get_factor_group_log_cumsum(self):
        """
        累计对数收益
        """
        factor_group_log_returns = np.log(1 + self.get_factor_group_returns())
        excess_long_log_returns = (
            factor_group_log_returns[self.n_quantiles]
            - factor_group_log_returns["passive"]
        )
        excess_short_log_returns = (
            factor_group_log_returns[1] - factor_group_log_returns["passive"]
        )
        excess_long_short_log_returns = (
            factor_group_log_returns[self.n_quantiles]
            - factor_group_log_returns[1]
            - factor_group_log_returns["passive"]
        )
        excess_log_cumsum = pd.DataFrame(
            {
                "long_excess": excess_long_log_returns,
                "short_excess": excess_short_log_returns,
                "long_short": excess_long_short_log_returns,
            },
            index=excess_long_log_returns.index,
        )

        excess_log_cumsum = excess_log_cumsum.cumsum()

        return excess_log_cumsum

    def get_factor_long_net_value(self, is_log=True):
        """
        多头净值曲线
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[self.n_quantiles]
        net_value = (returns + 1).cumprod()
        log_net_value = np.log(net_value) + 1 if is_log else net_value
        return log_net_value

    def get_factor_short_net_value(self, is_log=True):
        """
        空头净值曲线
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[1]
        net_value = (returns + 1).cumprod()
        log_net_value = np.log(net_value) + 1 if is_log else net_value
        return log_net_value

    def get_factor_long_short_net_value(self, is_log=True):
        """
        多空对冲净值曲线
        """
        factor_group_returns = self.get_factor_group_returns()
        returns = factor_group_returns[self.n_quantiles] - factor_group_returns[1]
        net_value = (returns + 1).cumprod()
        log_net_value = np.log(net_value) + 1 if is_log else net_value
        return log_net_value

    def get_factor_long_returns_log_cumsum(self):
        """
        多头累计对数收益曲线
        """
        factor_group_log_returns = np.log(1 + self.get_factor_group_returns())
        excess_log_returns = (
            factor_group_log_returns[self.n_quantiles]
            - factor_group_log_returns["passive"]
        )
        excess_log_cumsum = (excess_log_returns).cumsum()
        return excess_log_cumsum

    def get_factor_short_returns_log_cumsum(self):
        """
        空头累计对数收益曲线
        """
        factor_group_log_returns = np.log(1 + self.get_factor_group_returns())
        excess_log_returns = (
            factor_group_log_returns[1] - factor_group_log_returns["passive"]
        )
        excess_log_cumsum = (excess_log_returns).cumsum()
        return excess_log_cumsum

    def get_factor_long_short_returns_log_cumsum(self):
        """
        空头累计对数收益曲线
        """
        factor_group_log_returns = np.log(1 + self.get_factor_group_returns())
        excess_log_returns = (
            factor_group_log_returns[self.n_quantiles]
            - factor_group_log_returns[1]
            - factor_group_log_returns["passive"]
        )
        excess_log_cumsum = (excess_log_returns).cumsum()
        return excess_log_cumsum

    def get_max_drawdown_series(self):
        cumulative_returns_series = self.get_factor_long_net_value(is_log=False)
        running_max = cumulative_returns_series.cummax()
        drawdown_series = cumulative_returns_series / running_max - 1
        drawdown_series[drawdown_series >= 0] = 0
        return drawdown_series

    def get_group_mean_returns(self):
        factor_group_returns = self.get_factor_group_returns().drop(columns="passive")
        group_mean_returns = factor_group_returns.sub(
            factor_group_returns.mean(axis=1), axis=0
        ).mean(axis=0)
        return group_mean_returns

    def get_ic_series(self, method="spearman"):
        """
        Calculate the Information Cioefficient (IC) series for the factor.

        Parameters:
        - method: Method used for correlation calculation. Default is 'spearman'.

        Returns:
        - monthly_ic: A pandas Series containing the monthly IC values.
        """

        # Calculate monthly IC
        ic_series = self.factor.corrwith(self.future_returns, method=method, axis=1)

        return ic_series

    def get_indicators(self):
        # 1. get indicators of the factor, about the IC
        ic_series = self.get_ic_series("spearman")

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_skew = ic_series.skew()
        ic_kurt = ic_series.kurt()
        icir = (ic_mean / ic_std) * np.sqrt(self.period)
        t = ic_series.mean() * np.sqrt(len(ic_series.dropna())) / ic_series.std()
        ic_gt_0 = len(ic_series[ic_series > 0]) / len(ic_series[ic_series.notna()])

        # 2. get indicators of factor group returns

        long_returns = self.get_factor_long_returns(is_log=False)
        short_returns = -self.get_factor_short_returns(is_log=False)
        long_short_returns = self.get_factor_long_short_returns(is_log=False)

        # bm_ret = align_series(get_benchmark_return(),long_short_returns.index)

        ## Annualized Return by empyrical
        long_ann_ret = empyrical.annual_return(long_returns, annualization=self.period)
        short_ann_ret = empyrical.annual_return(
            short_returns, annualization=self.period
        )
        long_short_ann_ret = empyrical.annual_return(
            long_short_returns, annualization=self.period
        )
        benchmar_ann_ret = empyrical.annual_return(
            self.benchmark_returns, annualization=self.period
        )

        ## MaxDrawdown
        net_value = self.get_factor_long_net_value(is_log=False)
        running_max = net_value.cummax()
        drawdown_series = 1 - net_value / running_max
        max_drawdown = drawdown_series.max()
        max_drawdown_end = drawdown_series.idxmax()
        max_drawdown_start = net_value[:max_drawdown_end].idxmax()
        max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
        max_drawdown_end = drawdown_series.idxmax().strftime("%Y-%m-%d")
        max_drawdown_start = net_value[:max_drawdown_end].idxmax().strftime("%Y-%m-%d")

        # Stats
        long_sharpe = empyrical.sharpe_ratio(long_returns, annualization=self.period)
        long_calmar = empyrical.calmar_ratio(long_returns, annualization=self.period)
        short_sharpe = empyrical.sharpe_ratio(short_returns, annualization=self.period)
        short_calmar = empyrical.calmar_ratio(short_returns, annualization=self.period)
        long_short_sharpe = empyrical.sharpe_ratio(
            long_short_returns, annualization=self.period
        )
        long_short_calmar = empyrical.calmar_ratio(
            long_short_returns, annualization=self.period
        )
        pre_ret = (
            self.get_factor_long_returns() - self.get_factor_group_returns()["passive"]
        )
        info_ratio = empyrical.sharpe_ratio(pre_ret, annualization=self.period)
        # Volatility
        long_ann_vol = empyrical.annual_volatility(
            long_returns, annualization=self.period
        )
        short_ann_vol = empyrical.annual_volatility(
            short_returns, annualization=self.period
        )
        long_short_ann_vol = empyrical.annual_volatility(
            long_short_returns, annualization=self.period
        )

        # Win Rate
        win_rate = len(long_returns[long_returns > 0]) / len(
            long_returns[long_returns.notna()]
        )
        win_excess_rate = len(
            long_returns[long_returns > self.benchmark_returns]
        ) / len(long_returns[long_returns.notna()])

        # 3. get indicators of factor turnover rate

        short_turnover, long_turnover = self.get_factor_turnover_rate()

        return {
            "info_ratio": info_ratio,
            "long_ret": long_ann_ret,
            "short_ret": short_ann_ret,
            "long_short_ret": long_short_ann_ret,
            "benchmark_ret": benchmar_ann_ret,
            "long_sharpe": long_sharpe,
            "long_calmar": long_sharpe,
            "short_sharpe": short_sharpe,
            "short_calmar": short_calmar,
            "long_short_sharpe": long_short_sharpe,
            "long_short_calmar": long_short_calmar,
            "short_turnover": short_turnover,
            "long_turnover": long_turnover,
            "win_rate": win_rate,
            "win_excess_rate": win_excess_rate,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_skew": ic_skew,
            "ic_kurt": ic_kurt,
            "icir": icir,
            "t": t,
            "ic_gt_0": ic_gt_0,
            "max_drawdown": max_drawdown,
            "max_drawdown_start": max_drawdown_start,
            "max_drawdown_end": max_drawdown_end,
            "max_drawdown_duration": max_drawdown_duration,
            "long_vol": long_ann_vol,
            "short_vol": short_ann_vol,
            "long_short_vol": long_short_ann_vol,
        }

    def get_factor_turnover_rate(self):
        # Filter stocks based on stock pool
        self.factor.index.name = "tradedate"

        turnover_list = []

        # 只计算第一组和最后一组的换手率
        for i in range(1, self.n_quantiles + 1):
            if i == 1 or i == self.n_quantiles:
                factor_group = (
                    self.quantiled_factor[self.quantiled_factor == i]
                    .notna()
                    .astype(int)
                )
                turnover = 0
                for date in factor_group.index:
                    if date == factor_group.index[0]:
                        continue
                    else:
                        current_hold = factor_group.loc[date]
                        last_hold = factor_group.shift(1).loc[date]
                        turnover = (
                            turnover
                            + current_hold.ne(last_hold).sum() / current_hold.sum()
                        )
                turnover = turnover / (len(factor_group.index) - 1) * self.period
                turnover_list.append(turnover)

        short_turnover, long_turnover = turnover_list
        return short_turnover, long_turnover

    def get_last_factor_value(self):
        factor = load_object_from_pickle_with_path(self.factor_name, self.target_dir)
        factor_adj = (
            factor.where(self.stock_pool == 1)
            .iloc[-1]
            .dropna()
            .sort_values(ascending=False)
        )
        return factor_adj

    def get_recent_data_for_viz(self):
        # 最近一期的因子值
        factor_series = self.get_last_factor_value()
        factor = factor_series.to_frame("factor")
        factor_rank = factor_series.rank(pct=True).to_frame("rank")
        viz_df = pd.concat([factor, factor_rank], axis=1)
        viz_df = viz_df.sort_values("rank", ascending=False)
        viz_df = viz_df.reset_index()

        info = load_object_from_pickle(f"basic_info_{self.prefix}")
        sesname_dict = dict(zip(info["symbol"], info["sesname"]))
        viz_df["sesname"] = viz_df.symbol.map(sesname_dict)
        if self.prefix == "stock":
            swlevel1_dict = dict(zip(info["symbol"], info["swlevel1name"]))
            swlevel2_dict = dict(zip(info["symbol"], info["swlevel2name"]))

            viz_df["swlevel1name"] = viz_df.symbol.map(swlevel1_dict)
            viz_df["swlevel2name"] = viz_df.symbol.map(swlevel2_dict)
            viz_df = viz_df[
                [
                    "sesname",
                    "symbol",
                    "swlevel1name",
                    "swlevel2name",
                    "factor",
                    "rank",
                ]
            ]
        else:
            viz_df = viz_df[["sesname", "symbol", "factor", "rank"]]

        return viz_df

    def ic_plot(self, figsize=(20, 5), rolling_window=12):
        """Plot Information Coefficient (IC) time series and its rolling mean."""
        # Retrieve IC series and format index for plotting
        ic = self.get_ic_series()
        ic.index = ic.index.strftime("%Y-%m-%d")

        # Create rolling mean of IC and plot
        plt.figure(figsize=figsize)
        ic.rolling(rolling_window).mean().plot()
        ic.plot(kind="bar", title=f"{self.factor_name} Time Series IC")

        # Set x-ticks to be quarterly intervals
        plt.xticks(np.arange(0, len(ic), step=10), ic.index[::10], rotation=45)
        plt.show()

    def bin_plot(self, kind="bar", title=None):
        """Plot bin or group returns."""
        title = title or f"{self.factor_name} Group Returns"
        self.get_group_mean_returns().plot(kind=kind, title=title)
        plt.show()

    def net_val_plot(self, figsize=(10, 4)):
        """Plot net value of factor groups."""
        self.get_factor_group_net_value().plot(
            figsize=figsize,
            title=f"{self.factor_name} : Factor Group Net Value",
        )
        plt.show()

    def ls_plot(self, figsize=(6, 3)):
        """Plot Long-Short Net Value."""
        net_val = (
            self.get_factor_long_net_value()
            - self.get_factor_short_net_value()
        )
        net_val.plot(
            figsize=figsize,
            title=f"{self.factor_name} Factor Long-Short Net Value",
            secondary_y=True,
        )
        plt.show()

    def long_plot(self, figsize=(15, 6), ylim=(-1, 0)):
        """Plot Long Factor Group Returns along with Passive and Max Drawdown."""
        group_net_val = self.get_factor_group_net_value(demeaned=False)
        group_net_val[self.n_quantiles].plot(
            figsize=figsize, title=f"{self.factor_name} Factor Group Returns"
        )
        group_net_val['passive'].plot(figsize=figsize)
        
        max_drawdown = self.get_max_drawdown_series()
        max_drawdown.plot(
            figsize=figsize, label="Max Drawdown", secondary_y=True, ylim=ylim
        )
        plt.legend()
        plt.show()

    def plot_all(self, figsize=(20, 15)):
        """Plot all analytical graphs using subplots."""
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)  # Adjust the grid size based on the number of plots

        # Plot 1: Information Coefficient (IC)
        ic = self.get_ic_series()
        ic.index = ic.index.strftime('%Y-%m-%d')
        ic.rolling(12).mean().plot(ax=axes[0, 0], title=f'IC Rolling Mean')
        axes[0, 0].bar(ic.index, ic.values, color='grey')
        axes[0, 0].set_xticks([])  # Remove x-ticks for clarity

        # Plot 2: Bin Plot
        self.get_group_mean_returns().plot(kind='bar', ax=axes[0, 1], title=f'Group Returns')

        # Plot 3: Net Value Plot
        self.get_factor_group_net_value().plot(ax=axes[1, 0], title=f'Factor Group Net Value')
        axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        # Plot 4: Long-Short Net Value Plot
        ls_val = self.get_factor_long_net_value() - self.get_factor_short_net_value()
        ls_val.plot(ax=axes[1, 1], title=f'LS Net Value')

        # Plot 5: Long Factor Group Returns
        group_net_val = self.get_factor_group_net_value(demeaned=False)
        group_net_val[self.n_quantiles].plot(ax=axes[2, 0], title=f'Long Factor Group Returns')
        group_net_val['passive'].plot(ax=axes[2, 0])
        
        # Plot 6: Max Drawdown Series
        max_drawdown = self.get_max_drawdown_series()
        max_drawdown.plot(ax=axes[2, 1], title='Max Drawdown', color='red')
        axes[2, 1].fill_between(max_drawdown.index, max_drawdown, color='red', alpha=0.3)

        # Adjusting the layout, titles, and x-axis labels
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the subplot layout

        # Adding a main title
        plt.suptitle(f'Analysis Plots', fontsize=16)
        # Show the plots
        plt.show()


if __name__ == "__main__":
    index_estclass = IndexEstclass.SWLEVEL1_INDEX

    params = {
        "prefix": "index",
        "index_estclass": index_estclass,
        "freq": "M",
        "n_quantiles": 5,
        "start": "2022-01-01",
        "end": "2024-01-01",
        "ascending": True,
        "benchmark_symbol": "000300",
        "scope": None,
        "commission_rate": 0.000,
        "slippage_rate": 0.000,
    }
    analyzer = Analyzer(**params)
    analyzer.set_target_factor("s_corr")
