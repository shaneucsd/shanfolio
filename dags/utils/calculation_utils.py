import numpy as np
import pandas as pd


def calculate_listed_days(series: pd.Series):
    first_valid_index = series.first_valid_index()
    if first_valid_index is not None:
        return (series.index - first_valid_index).days + 1
    else:
        return pd.Series([np.nan] * len(series), index=series.index)


def get_amount_over_limit(amount_of_a_index: pd.DataFrame):
    # 计算每个交易日的总成交额
    total_amount = amount_of_a_index.sum(axis=1)

    # 创建一个矩阵，用于标记选择的个股
    selected_stocks = pd.DataFrame(0, index=amount_of_a_index.index, columns=amount_of_a_index.columns)

    # 遍历每个交易日
    for date in amount_of_a_index.index:
        # 按成交额降序排序
        sorted_stocks = amount_of_a_index.loc[date].sort_values(ascending=False)

        # 初始化已选择的成交额之和
        selected_amount = 0

        # 逐个选择个股，直到成交额之和达到总成交额的99%
        for symbol, stock_amount in sorted_stocks.items():
            selected_stocks.loc[date, symbol] = 1  # 标记为1表示选择
            selected_amount += stock_amount

            if selected_amount >= total_amount[date] * 0.99:
                break
    selected_stocks = selected_stocks.replace({0: np.nan})
    return selected_stocks

