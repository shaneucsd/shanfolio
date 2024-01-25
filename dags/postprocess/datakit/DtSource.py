import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from enums.data_dimension import DataDimension
from utils.file_utils import FACTOR_PATH, SQL_TASKS_PATH, read_pickle_from_sql_tasks
from utils.financials_utils import FinancialDtCleaner

START_DATE = '2009-01-01'
END_DATE = '2022-06-30'
EXCHANGE = 'ASHARE'
index_comp_path = 'airflow/output/sql_tasks/info_index_comp.pickle'

TODAY = datetime.datetime.now().strftime("%Y%m%d")


class DtSource:
    """
    Default reading Method is Parquet

    Data Include:
        - SK data daily

    """

    def __init__(self, prefix='stock'):

        self.path = SQL_TASKS_PATH
        self.prefix = prefix

    def name_modify(self, f_name: Path):
        if f_name.suffix == '.pickle':
            f_name = f_name.stem
        return f_name.lower()

    # Read Single File
    def read_file(self, path, filename):
        file_path = Path(path) / filename
        ext = file_path.suffix.lower()

        if ext == '.pickle':
            df = pd.read_pickle(file_path)
            df = self.format_df(df)
        else:
            msg = f'Unsupported file type: {ext}, of {file_path}'
            raise TypeError(msg)

        return df

    def read_multiple_files(self, items):
        def _yield_df(_items):
            for i in range(len(_items)):
                path, filename = _items[i][0], _items[i][1]
                df = self.read_file(path, filename)
                yield df

        dfs = [df for df in _yield_df(items)]
        assert len(dfs) > 0, 'No Data in Directory, {items}'
        df = pd.concat(dfs)
        return df

    def format_df(self, df):
        def convert_symbol(x):
            # if x can be converted to int
            try:
                x = str(int(x)).zfill(6)
            except:
                pass
            return x

        # Lower columns name
        df.columns = df.columns.str.lower()

        # Remove Unnamed Column
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # 如果列名以date结尾，则处理date
        for col in df.columns:
            if col.lower().endswith('date'):
                # 转换为日期对象
                df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')

        # Remove Duplicated Rows
        df = df[df['symbol'].notna()]

        df.reset_index(inplace=True)

        if 'index' in df.columns:
            df.drop('index', axis=1, inplace=True)

        # Format Symbol
        if EXCHANGE == 'ASHARE':
            df['symbol'] = df['symbol'].apply(lambda x: convert_symbol(x))

        # CRITICAL ERROR HERE:
        # format的convert_symbol之后，出现了重复值，猜测可能是有些symbol存储为int，convert后出现重复值
        # 为什么会出现字符串呢？ 因为forecast的sql是用navicat跑的，airflow上太大，本地没有sqlserver的odbc，是手动存为csv，此时字符串为int， 然后再python转为pickle的。
        # 可行的方法： 再本地装sqlserver的操作。
        # 为了保证不要出问题，
        
        df.drop_duplicates(subset=['symbol', 'tradedate'], inplace=True)

        return df

    def _pivot_and_save(self, df):

        def _pivot_col(df, col):
            pivot_df = df.pivot(index='tradedate', columns='symbol', values=col)
            return pivot_df

        def _save(df, file_path):
            df = df[~df.index.duplicated(keep='last')]
            df.to_pickle(file_path)

            return df

        print('Data From: ', df.tradedate.min(), ' to ', df.tradedate.max())

        target_cols = []
        for col in df.columns:
            # :TODO 直接尝试将columns转为
            if df[col].dtype in ['float64', 'int64'] and col not in ['tradedate', 'symbol', 'secode', 'index',
                                                                     'publishdate', 'begindate', 'enddate',
                                                                     'reportdatetype']:
                target_cols.append(col)

        print('    Target cols: ', target_cols)
        # 例如stock_sk，只取下划线前第一个元素，作为文件夹名
        dir_path = Path(FACTOR_PATH) / self.prefix.split("_")[0]

        if not dir_path.exists():
            dir_path.mkdir()

        # Recheck if duplicated exists.


        for col in target_cols:
            file_path = Path(dir_path) / f'{col}.pkl'
            pivot_df = _pivot_col(df, col)
            _save(pivot_df, file_path)
            print('    Col saved: ', col)

    def _insert(self, items):
        for path, filename in items:
            if 'curr' in filename:
                df = self.read_file(path, filename)
                return df

    def read_data_and_pivot_to_pickle(self, tag: DataDimension):

        # Stock
        if tag == DataDimension.STOCK_SK_MARKET:
            df = read_pickle_from_sql_tasks('stock_sk_market')
        elif tag == DataDimension.STOCK_SK_MONEYFLOW:
            df = read_pickle_from_sql_tasks('stock_sk_moneyflow')
        elif tag == DataDimension.STOCK_FORECAST:
            df = read_pickle_from_sql_tasks('stock_sk_forecast')
        elif tag == DataDimension.STOCK_VALUATION:
            df = read_pickle_from_sql_tasks('stock_valuation')
        elif tag == DataDimension.STOCK_SHHK:
            df = read_pickle_from_sql_tasks('stock_shhk_northamt')
        elif tag == DataDimension.FINANCIALS_BALANCE:
            df = read_pickle_from_sql_tasks('financials_balance')
            fin_dtcleaner = FinancialDtCleaner(df)
            df = fin_dtcleaner.get_balance_df()
        elif tag == DataDimension.FINANCIALS_INCOME:
            df = read_pickle_from_sql_tasks('financials_income')
            fin_dtcleaner = FinancialDtCleaner(df)
            ttm_df = fin_dtcleaner.get_ttm_df()
            mrq_df = fin_dtcleaner.get_mrq_df()
            df = [ttm_df, mrq_df]
        elif tag == DataDimension.FINANCIALS_CASH_FLOW:
            df = read_pickle_from_sql_tasks('financials_cash_flow')
            fin_dtcleaner = FinancialDtCleaner(df)
            ttm_df = fin_dtcleaner.get_ttm_df()
            mrq_df = fin_dtcleaner.get_mrq_df()
            df = [ttm_df, mrq_df]
        # Index
        elif tag == DataDimension.INDEX_SK_MARKET:
            df = read_pickle_from_sql_tasks('index_sk_market')
        else:
            raise NotImplementedError("请自行添加新的类型")

        print(f'{tag}: read, pivot and saving ...')
        if isinstance(df, pd.DataFrame):
            df = self.format_df(df)
            self._pivot_and_save(df)

        elif isinstance(df, list):
            for _df in df:
                _df = self.format_df(_df)
                self._pivot_and_save(_df)
        print(f'{tag}: saved ...')


# class WindIndexAppender():
#     wind_index_dict = {
#         '8841415.WI': '茅指数',  # 茅台指数
#         '8841447.WI': '宁指数',  # 宁德时代
#         '8841660.WI': 'AIGC',  # AIGC
#         'CI009187.WI': '复合集流体',  # 复合集流体,
#         '8841681.WI': '中特估',
#         '8841422.WI': '顺周期',
#         '8841241.WI': '存储器指数',
#         '8841678.WI': '算力指数',
#         '884201.WI': '人工智能',
#         '8841418.WI': '医美指数',
#
#     }
#
#
#     def __init__(self):
#
#         w.start()
#         self.tradeday = self.get_last_tradeday()
#         print('wind_index_dict', self.wind_index_dict)
#
#     def get_last_tradeday(self):
#         # get today datetime
#         today = datetime.datetime.today().strftime('%Y-%m-%d')
#         # get the first day of this month
#         first_day_of_this_month = datetime.datetime.today().replace(day=1).strftime('%Y-%m-%d')
#
#         e, tradedays = w.tdays(first_day_of_this_month, today, "", usedf=True)
#         tradeday = tradedays.index[0].strftime('%Y-%m-%d')
#         return tradeday
#
#     def get_sk_daily_df_from_wind(self, symbol, start, end):
#         e, wind_sk_daily_df = w.wsd(symbol, "close,low,high,open,volume,amt,pct_chg,val_mvc,val_mv_ARD", start, end,
#                                     "unit=1;PriceAdj=B", usedf=True)
#         wind_sk_daily_df = self.format_wind_sk_daily_df(symbol, wind_sk_daily_df)
#
#         return wind_sk_daily_df
#
#     def get_index_comp_from_wind(self, symbol):
#         e, wind_index_comp_df = w.wset("sectorconstituent", f"date={self.tradeday};windcode={symbol}", usedf=True)
#         wind_index_comp_df = self.format_wind_index_comp_df(symbol, wind_index_comp_df)
#         return wind_index_comp_df
#
#     def format_wind_index_comp_df(self, symbol, wind_index_comp_df):
#
#         wind_index_comp_df = wind_index_comp_df.rename(
#             columns={'wind_code': 'selectesymbol', 'sec_name': 'selectesesname'})
#         wind_index_comp_df['secode'] = np.nan
#         wind_index_comp_df['symbol'] = symbol
#         wind_index_comp_df['sesname'] = self.wind_index_dict[symbol]
#         wind_index_comp_df['selectesymbol'] = wind_index_comp_df['selectesymbol'].apply(lambda x: x[:6])
#         if 'date' in wind_index_comp_df.columns:
#             wind_index_comp_df = wind_index_comp_df.drop(columns='date')
#         wind_index_comp_df = wind_index_comp_df[['secode', 'symbol', 'selectesymbol', 'selectesesname', 'sesname']]
#
#         return wind_index_comp_df
#
#     def format_wind_sk_daily_df(self, symbol, wind_sk_daily_df):
#
#         wind_sk_daily_df = wind_sk_daily_df.reset_index()
#         wind_sk_daily_df['SYMBOL'] = symbol
#         wind_sk_daily_df['ESTCLASS'] = '中证主题指数'
#         wind_sk_daily_df['SESNAME'] = self.wind_index_dict[symbol]
#         wind_sk_daily_df['SECODE'] = None
#         wind_sk_daily_df['ENTRYDATE'] = None
#         wind_sk_daily_df['ENTRYTIME'] = None
#
#         wind_sk_daily_df = wind_sk_daily_df.rename(
#             columns={'index': 'TRADEDATE', 'AMT': 'AMOUNT', 'PCT_CHG': 'PCHG', 'VAL_MVC': 'NEGOTIABLEMV',
#                      'VAL_MV_ARD': 'TOTMKTCAP'})
#         wind_sk_daily_df = wind_sk_daily_df[['TRADEDATE', 'SECODE', 'SYMBOL', 'ESTCLASS', 'SESNAME', 'OPEN', 'CLOSE',
#                                              'HIGH', 'LOW', 'VOLUME', 'AMOUNT', 'PCHG', 'NEGOTIABLEMV', 'TOTMKTCAP',
#                                              'ENTRYDATE', 'ENTRYTIME']]
#
#         wind_sk_daily_df['TRADEDATE'] = wind_sk_daily_df['TRADEDATE'].apply(lambda x: x.strftime('%Y%m%d'))
#
#         return wind_sk_daily_df
#
#     def read_local_index_comp_df(self):
#         def convert_symbol(x):
#             # if x can be converted to int
#             try:
#                 x = str(int(x)).zfill(6)
#             except:
#                 pass
#             return x
#
#         # Index Compontent
#         index_comp_df = pd.read_csv(index_comp_path)
#         index_comp_df.columns = index_comp_df.columns.str.lower()
#         index_comp_df['symbol'] = index_comp_df['symbol'].apply(lambda x: convert_symbol(x))
#         index_comp_df['selectesymbol'] = index_comp_df['selectesymbol'].apply(lambda x: convert_symbol(x))
#
#         return index_comp_df
#
#     def append_to_local_index_comp_df(self):
#
#         index_comp_df = self.read_local_index_comp_df()
#
#         for symbol in self.wind_index_dict.keys():
#             wind_index_comp_df = self.get_index_comp_from_wind(symbol)
#             len_1 = len(wind_index_comp_df)
#             index_comp_df = index_comp_df.append(wind_index_comp_df)
#             index_comp_df.drop_duplicates(inplace=True)
#
#         index_comp_df.to_csv(index_comp_path, index=False)
#
#     def append_to_local_sk_daily_df(self, path):
#         print(path)
#         df = pd.read_csv(path)
#         tradedays = pd.Series(pd.to_datetime(df['TRADEDATE'], format='%Y%m%d').sort_values().unique()).to_list()
#
#         for symbol in self.wind_index_dict.keys():
#             curr_len = len(df)
#             wind_sk_daily_df = self.get_sk_daily_df_from_wind(symbol, tradedays[0], tradedays[-1])
#             df = df.append(wind_sk_daily_df)
#             df.drop_duplicates(inplace=True)
#         df.to_csv(path, index=False)
#         return df


# class GeneralLoader():
#
#     def __init__(self, ori_root=None, save_root=None):
#         self.ori_root = ori_root
#         self.save_root = save_root
#
#     def _get_index_comp(self, path='../source/index/index_comp/index_comp.csv'):
#         sec_index_comp_df = pd.read_csv(path)
#         # lower all columns
#         sec_index_comp_df.columns = sec_index_comp_df.columns.str.lower()
#         # fill symbol with 0
#         sec_index_comp_df['symbol'] = sec_index_comp_df['symbol'].apply(lambda x: str(x).zfill(6))
#         sec_index_comp_df['selectesymbol'] = sec_index_comp_df['selectesymbol'].apply(lambda x: str(x).zfill(6))
#
#         sec_index_comp_dict = {}
#         for symbol, group in sec_index_comp_df.groupby('symbol')['selectesymbol']:
#             sec_index_comp_dict.update({symbol: group.tolist()})
#             assert len(group.tolist()) > 0, 'No data for {}'.format(symbol)
#
#         return sec_index_comp_dict
#
#     def _get_risk_free_rate(self, path='../source/macro/国债到期收益率_10年.xlsx'):
#         risk_free_rate = pd.read_excel(path, parse_dates=True, index_col=[0])[4:]
#         risk_free_rate.index = pd.to_datetime(risk_free_rate.index)
#         risk_free_rate.index.name = 'tradedate'
#         risk_free_rate = (1 + risk_free_rate[['国债到期收益率:10年']] / 100) ** (1 / 252) - 1  # 转化为日收益率
#         return risk_free_rate


def pivot_col(df, col):
    df = pd.pivot(df, columns='symbol', values=col)
    df = df.replace(0, np.nan)
    df.ffill(inplace=True)
    return df


def dict_to_df(_dict):
    df = pd.DataFrame(_dict, index=['sesname']).T
    df.index.name = 'symbol'
    df.reset_index(inplace=True)
    return df


# check the modified time of a file
def get_file_modify_time(file_path):
    path_obj = Path(file_path)
    t = path_obj.stat().st_mtime
    # format to string
    s = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    return s


def get_index_info_comp_df_from_local(info_path, comp_path):
    def convert_symbol(x):
        # if x can be converted to int
        try:
            x = str(int(x)).zfill(6)
        except:
            pass
        return x

    index_info_df = pd.read_pickle(info_path)
    index_info_df.columns = index_info_df.columns.str.lower()
    index_info_df['symbol'] = index_info_df['symbol'].apply(lambda x: convert_symbol(x))

    index_comp_df = pd.read_pickle(comp_path)

    index_comp_df.columns = index_comp_df.columns.str.lower()
    index_comp_df['symbol'] = index_comp_df['symbol'].apply(lambda x: convert_symbol(x))
    index_comp_df['selectesymbol'] = index_comp_df['selectesymbol'].apply(lambda x: convert_symbol(x))

    return index_info_df, index_comp_df


if __name__ == '__main__':
    pass
