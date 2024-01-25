import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

EXCLUDE_COLUMNS = ["SYMBOL", "PUBLiSHDATE", "BEGINDATE", "ENDDATE", "REPORTDATETYPE"]


# Cleaning Functions
def format_df(df):
    columns_to_drop = [col for col in df.columns if "unname" in col.lower()]
    df.drop(columns=columns_to_drop, inplace=True)
    df.columns = df.columns.str.lower()
    df["reportdatetype"] = df["reportdatetype"].astype("str")
    df["symbol"] = df["symbol"].apply(lambda x: str(x).zfill(6))
    df["enddate"] = pd.to_datetime(df["enddate"], format="%Y%m%d")
    df["publishdate"] = pd.to_datetime(df["publishdate"], format="%Y%m%d")
    df.sort_values(by=["symbol", "enddate"], inplace=True)
    return df


def adjust_publish_date(df):
    # caihui会调整并使用最新的财务报告，所以publishdate有时会晚于规定的发布日期
    # 由于没有原始的财报，直接使用调正后的财报
    # 将所有晚于规定发布日期，修改至规定发布日期

    correct_rows = (
        (
            (df["enddate"].dt.quarter == 1)
            & (df["publishdate"].dt.month <= 4)
            & (df["publishdate"].dt.year == df["enddate"].dt.year)
        )
        | (
            (df["enddate"].dt.quarter == 2)
            & (df["publishdate"].dt.month <= 8)
            & (df["publishdate"].dt.year == df["enddate"].dt.year)
        )
        | (
            (df["enddate"].dt.quarter == 3)
            & (df["publishdate"].dt.month <= 10)
            & (df["publishdate"].dt.year == df["enddate"].dt.year)
        )
        | (
            (df["enddate"].dt.quarter == 4)
            & (df["publishdate"].dt.month <= 4)
            & (df["publishdate"].dt.year == df["enddate"].dt.year + 1)
        )
    )
    error_rows = ~correct_rows

    df.loc[
        error_rows & (df["enddate"].dt.quarter == 1), "publishdate"
    ] = pd.to_datetime(
        df.loc[error_rows & (df["enddate"].dt.quarter == 1), "enddate"].dt.year,
        format="%Y",
    ) + pd.DateOffset(
        months=3, day=30
    )
    df.loc[
        error_rows & (df["enddate"].dt.quarter == 2), "publishdate"
    ] = pd.to_datetime(
        df.loc[error_rows & (df["enddate"].dt.quarter == 2), "enddate"].dt.year,
        format="%Y",
    ) + pd.DateOffset(
        months=7, day=30
    )
    df.loc[
        error_rows & (df["enddate"].dt.quarter == 3), "publishdate"
    ] = pd.to_datetime(
        df.loc[error_rows & (df["enddate"].dt.quarter == 3), "enddate"].dt.year,
        format="%Y",
    ) + pd.DateOffset(
        months=9, day=30
    )
    df.loc[
        error_rows & (df["enddate"].dt.quarter == 4), "publishdate"
    ] = pd.to_datetime(
        df.loc[error_rows & (df["enddate"].dt.quarter == 4), "enddate"].dt.year,
        format="%Y",
    ) + pd.DateOffset(
        years=1, months=3, day=30
    )
    return df


def ensure_continuity(df):
    """
    # Ensure there is continuous financial report for each symbol
    # Insert null financial report for missing reports

    :param df: 'symbol', 'publishdate', 'enddate', 'reportdatetype', 'totasset',...
    :return:  'symbol', 'publishdate', 'enddate', 'reportdatetype', 'totasset',...
    """

    def _ensure_continuity_group(_group):
        # Determining the start and end dates from the existing data
        start_date = _group["enddate"].min()
        end_date = _group["enddate"].max()

        # Defining the end dates for each quarter using the existing data range
        expected_quarter_end_dates = pd.date_range(
            start=start_date, end=end_date, freq="Q"
        )

        # Dictionary to map quarter to the corresponding publish date deadline
        publish_date_deadlines = {
            1: pd.DateOffset(months=4, days=-1),  # 4.30 for Q1
            2: pd.DateOffset(months=8, days=-1),  # 8.30 for Q2
            3: pd.DateOffset(months=10, days=-1),  # 10.30 for Q3
            4: pd.DateOffset(months=16, days=-1),  # 4.30 next year for Q4
        }

        missing_rows = []
        for end_date in expected_quarter_end_dates:
            if end_date not in _group["enddate"].values:
                quarter = (end_date.month - 1) // 3 + 1
                # Creating a row for the missing quarter
                missing_row = pd.DataFrame(
                    {
                        "symbol": _group["symbol"].iloc[0],
                        "publishdate": [end_date + publish_date_deadlines[quarter]],
                        "enddate": [end_date],
                        "reportdatetype": [str(quarter)],
                    }
                )
                missing_rows.append(missing_row)

        # Concatenating the missing rows with the original dataframe
        if missing_rows:
            _group = pd.concat([_group] + missing_rows, ignore_index=True)

        # Sorting the dataframe by enddate
        _group = _group.sort_values(by="enddate")
        return _group

    df = df.groupby("symbol").apply(lambda group: _ensure_continuity_group(group))
    df = df.droplevel(0)
    df = df.sort_values(["symbol", "enddate"]).reset_index(drop=True)

    return df


def clean_duplicates_financial_report(df):
    df.drop_duplicates(subset=["symbol", "enddate"], keep="last", inplace=True)
    return df


def validate_balance_df(df):
    #  粗糙的validate
    # 平安银行2022 06 30，总资产为5.108776e+12
    return (
        df.query('enddate=="2022-06-30" and symbol =="000001"')["totasset"].values[0]
        == 5.108776e12
    )


def validate_ttm_df(df):
    return (
        df.query('enddate=="2022-06-30" and symbol =="000001"')["bizinco_ttm"].values[0]
        == 176725000000.0
    )


# Converting Functions


def convert_to_mrq(df, numerical_columns):
    # 将累计值转为单季度的值
    #
    df["enddate_quarter"] = df["enddate"].dt.quarter
    enddate_quarter = df.pivot(
        index="enddate", columns="symbol", values="enddate_quarter"
    )

    for col in numerical_columns:
        if col == "enddate_quarter":
            continue
        col_df = df.pivot(index="enddate", columns="symbol", values=col)
        if col_df.dtypes.iloc[0] == np.dtype("float64"):
            col_df.loc[:, :] = np.where(col_df is None, np.nan, col_df)
            # 如果是一季报，直接使用一季报，否则使用季报与上一个季报的差值
            col_df = pd.DataFrame(
                np.where(enddate_quarter == 1, col_df, (col_df - col_df.shift(1))),
                index=enddate_quarter.index,
                columns=enddate_quarter.columns,
            )
            col_df = pd.melt(
                col_df.reset_index(),
                id_vars="enddate",
                value_vars=col_df.columns,
                value_name=col,
            )
            df = pd.merge(
                df.drop(col, axis=1), col_df, on=["enddate", "symbol"], how="left"
            )

    df.drop(columns=["enddate_quarter"], inplace=True)
    return df


def convert_to_ttm(df, numerical_columns):
    # 修改为TTM值

    df["enddate_quarter"] = df["enddate"].dt.quarter
    enddate_quarter = df.pivot(
        index="enddate", columns="symbol", values="enddate_quarter"
    )

    for col in numerical_columns:
        if col == "enddate_quarter":
            continue
        col_df = df.pivot(index="enddate", columns="symbol", values=col).astype(float)

        col_df_last_4q = np.where(
            enddate_quarter == 4,
            col_df.shift(4),
            np.where(
                enddate_quarter == 3,
                col_df.shift(3),
                np.where(enddate_quarter == 2, col_df.shift(2), col_df.shift(1)),
            ),
        )

        col_df_yoy = col_df.shift(4)

        my_col_df = np.where(
            ~np.isnan(col_df_yoy) & ~np.isnan(col_df_last_4q),
            col_df + col_df_last_4q - col_df_yoy,
            np.where(
                enddate_quarter == 4,
                col_df,
                np.where(
                    enddate_quarter == 3,
                    col_df * (4 / 3),
                    np.where(enddate_quarter == 2, col_df * (4 / 2), col_df * 4),
                ),
            ),
        )

        col_df = pd.DataFrame(
            my_col_df, index=enddate_quarter.index, columns=enddate_quarter.columns
        )
        col_df = pd.melt(
            col_df.reset_index(),
            id_vars="enddate",
            value_vars=col_df.columns,
            value_name=f"{col}_ttm",
        )
        df = pd.merge(
            df.drop(col, axis=1), col_df, on=["enddate", "symbol"], how="left"
        )

    df.drop(columns=["enddate_quarter"], inplace=True)
    return df


def create_latency_columns(df, numerical_columns, n=20):
    """
    为了保证能够快速、准确的读取到前1、2、3个季报和前一年的数据，创建Latency列

    col_1 : 1季度前的数据
    col_2 : 2季度前的数据
    col_3 : 3季度前的数据
    col_4 : 一年前的数据


    Args:
        df (_type_): _description_
        numerical_columns (_type_): 需要添加latency的列

    Returns:
        _type_: _description_

    """

    df = df.sort_values(by=["symbol", "enddate"])
    for col in numerical_columns:
        for i in range(1, n + 1):  # 创建前1、2、3个季度的数据
            latency_col_name = f"{col}_{i}"  # 新列的名称
            df[latency_col_name] = df.groupby("symbol")[col].shift(i)
    return df


# Post Cleaning Functions


def post_cleaning(df, clean_type="drop_q4_reports_if_q1_exists"):
    """

    这里有非常重要的问题要解决

    1. most recent financials，仅存储最新一期的财务数据（TTM或MRQ）
        - 计算简单，始终保留最新的财务数据
        - 会缺失四季度的数据（因为和一季报同时发布）
        - 问题：无法准确获得前n季度的财务数据，比如要计算最近5年20个季度的营收增长斜率。
    2. by quarter， 按照quarter 存储
        - TODO


    """
    if clean_type == "drop_q4_reports_if_q1_exists":
        df.drop_duplicates(subset=["symbol", "publishdate"], keep="last", inplace=True)
        df.rename(columns={"publishdate": "tradedate"}, inplace=True)
        return df


def preprocess(df):
    df = format_df(df)
    df = clean_duplicates_financial_report(df)
    df = adjust_publish_date(df)
    df = ensure_continuity(df)
    df.sort_values(by=["symbol", "enddate"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class FinancialDtCleaner:
    """
    Data Validation:
        - Format financials df
        - Adjust Records PublishDate
        - Convert Cumsum value to Single Quarter value, and TTM value
        - Assure Continuity of Records

    Financials DF:
            <class 'pandas.core.frame.DataFrame'>
                RangeIndex: 254984 entries, 0 to 254983
                Data columns (total 15 columns):
                 #   Column           Non-Null Count   Dtype
                ---  ------           --------------   -----
                 0   SYMBOL           254984 non-null  int64
                 1   PUBLISHDATE      254984 non-null  int64
                 2   ENDDATE          254984 non-null  int64
                 3   REPORTDATETYPE   254984 non-null  int64
                 4   TOTASSET         254956 non-null  float64
                 5   TOTCURRASSET     249231 non-null  float64
                 6   TOTALNONCASSETS  249436 non-null  float64
                 7   TOTLIAB          254414 non-null  float64
                 8   TOTALNONCLIAB    245268 non-null  float64
                 9   PREST            1992 non-null    float64
                 10  PERBOND          3745 non-null    float64
                 11  MINYSHARRIGH     196213 non-null  float64
                 12  TOTALCURRLIAB    249233 non-null  float64
                 13  PARESHARRIGH     254953 non-null  float64
                 14  RIGHAGGR         254892 non-null  float64
                dtypes: float64(11), int64(4)
                memory usage: 29.2 MB
    Examples:
        fin_dtcleaner = FinancialDtCleaner(df)
        mrq_df = self.get_ttm_df()
    """

    def __init__(self, df):
        self._df = df
        self._df = preprocess(self._df)
        self._num_cols = [
            col
            for col in df.columns
            if col.upper() not in map(str.upper, EXCLUDE_COLUMNS)
        ]

    def get_mrq_df(self):
        mrq_df = convert_to_mrq(self._df, self._num_cols)
        mrq_df = create_latency_columns(mrq_df, self._num_cols, n=4)
        mrq_df = post_cleaning(mrq_df, clean_type="drop_q4_reports_if_q1_exists")
        return mrq_df

    def get_ttm_df(self):
        ttm_df = convert_to_ttm(self._df, self._num_cols)
        ttm_df = post_cleaning(ttm_df, clean_type="drop_q4_reports_if_q1_exists")
        return ttm_df

    def get_balance_df(self):
        # Balance sheet needs no adjust.
        df = self._df
        df = create_latency_columns(df, self._num_cols, n=4)
        df = post_cleaning(df, clean_type="drop_q4_reports_if_q1_exists")
        return df
