import numpy as np
import pandas as pd
from utils.file_utils import load_object_from_pickle

tradedays = load_object_from_pickle("tradedays")


def get_benchmark_close(symbol="000985"):
    index_close = load_object_from_pickle("index_close")
    bm_close = index_close[symbol]

    # TODO: For testing usage, remove this line later
    # handle if not all tradedays are in bm_close
    if len(bm_close) != len(tradedays):
        bm_close = bm_close.reindex(tradedays)
    else:
        bm_close = bm_close.loc[tradedays]
    return bm_close


def get_benchmark_return(symbol="000985"):
    bm_close = get_benchmark_close(symbol)
    bm_ret = np.log(bm_close).diff()
    return bm_ret


def align_series(series, index):
    series = series.reindex(index)
    series = series.loc[index]
    return series


# Helper function to align data
def align_data_w_stock_pool(factor, price, benchmark_price, stock_pool=None, freq="M"):
    """
    Aligns the data without using pd.concat for better performance.

    Parameters:
    - factor: DataFrame of factor values.
    - price: DataFrame of price values.
    - stock_pool: DataFrame indicating if a stock is tradable (1) or not (0) for each day.
    - freq: Frequency for resampling. Default is 'M' for monthly.

    Returns:
    - aligned: A dictionary containing aligned factor, future_returns, and stock_pool.
    """
    if stock_pool is not None:
        factor = factor.where(stock_pool == 1)
        price = price.where(stock_pool == 1)

    factor.dropna(how="all", axis=0, inplace=True)
    price.dropna(how="all", axis=0, inplace=True)

    factor = factor.resample(freq).last()
    price = price.resample(freq).last()
    benchmark_price = benchmark_price.resample(freq).last()

    factor.dropna(how="all", axis=0, inplace=True)

    union_index = factor.index.intersection(price.index).intersection(
        benchmark_price.index
    )
    union_columns = factor.columns.intersection(price.columns)

    factor = factor.loc[union_index, union_columns]
    price = price.loc[union_index, union_columns]
    benchmark_price = benchmark_price.loc[union_index]

    future_returns = price.pct_change().shift(-1)
    benchmark_returns = benchmark_price.pct_change().shift(-1)

    return factor, price, future_returns, benchmark_returns


def align_data_wo_stock_pool(factor, price, benchmark_price, freq="M"):
    """
    Aligns the data without using pd.concat for better performance.

    Parameters:
    - factor: DataFrame of factor values.
    - price: DataFrame of price values.
    - stock_pool: DataFrame indicating if a stock is tradable (1) or not (0) for each day.
    - freq: Frequency for resampling. Default is 'M' for monthly.

    Returns:
    - aligned: A dictionary containing aligned factor, future_returns, and stock_pool.
    """
    common_dates = factor.index.intersection(price.index).intersection(
        benchmark_price.index
    )
    common_stocks = factor.columns.intersection(price.columns)

    # Resample and align
    aligned_factor = factor.loc[common_dates, common_stocks].resample(freq).last()
    aligned_price = price.loc[common_dates, common_stocks].resample(freq).last()
    aligned_benchmark_price = benchmark_price.loc[common_dates].resample(freq).last()
    future_returns, benchmark_returns = get_actual_returns(
        aligned_price, aligned_benchmark_price
    )

    # Fixing None
    aligned_factor = aligned_factor.where(aligned_factor.notnull(), np.nan)
    aligned_price = aligned_price.where(aligned_price.notnull(), np.nan)
    future_returns = aligned_price.where(aligned_price.notnull(), np.nan)
    aligned_benchmark_price = aligned_benchmark_price.where(
        aligned_benchmark_price.notnull(), np.nan
    )

    return aligned_factor, aligned_price, future_returns, benchmark_returns


def get_actual_returns(aligned_price, aligned_benchmark_price, is_log=False):
    future_returns = aligned_price / aligned_price.shift(1) - 1
    benchmark_returns = aligned_benchmark_price / aligned_benchmark_price.shift(1) - 1

    future_returns = future_returns.shift(-1)
    benchmark_returns = benchmark_returns.shift(-1)

    return future_returns, benchmark_returns

def datetime_to_int(dt):
    try:
        return int(dt.strftime('%Y%m%d'))
    except:
        return int(dt)
    
def get_stk_industry(end_dt,stk_list) -> pd.DataFrame:
    # 行业: 截止日的行业分类 // 中信行业

    # Index Components
    index_components  = load_object_from_pickle(tag='index_components')
    index_components.outdate = index_components.outdate.map(datetime_to_int)
    swlevel1_index_components = index_components.query('(estclass=="申万一级行业指数") & (usestatus=="1") & outdate > @end_dt')
    swlevel1_dict = dict(zip(swlevel1_index_components['selectedsymbol'],swlevel1_index_components['symbol']))

    swlevel3_index_components = index_components.query('(estclass=="申万三级行业指数") & (usestatus=="1") & outdate > @end_dt')
    swlevel3_dict = dict(zip(swlevel3_index_components['selectedsymbol'],swlevel3_index_components['symbol']))
   
    # 1. 创建一个新的DataFrame，索引为stk_list
    industry_df = pd.DataFrame(index=stk_list)

    # 2. 添加两列，一列用于一级行业代码，一列用于三级行业代码
    # 初始化这些列的值为'other'
    industry_df['level1'] = 0
    industry_df['level3'] =  0

    # 3. 遍历stk_list，使用字典来填充行业代码
    for stk in stk_list:
        industry_df.at[stk, 'level1'] = swlevel1_dict.get(stk, 0)
        industry_df.at[stk, 'level3'] = swlevel3_dict.get(stk, 0)

    return industry_df

