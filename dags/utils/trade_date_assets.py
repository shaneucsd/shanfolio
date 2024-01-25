import datetime
import numpy as np
import pandas as pd
import tushare as ts
from enums.index_estclass import IndexEstclass
from utils.basic_info import get_basic_info_for_index
from utils.file_utils import load_object_from_pickle, save_object_to_pickle
import warnings

warnings.filterwarnings('ignore')

token = "4de969c81ef5f589fed566bb5e3e89808d86c59eed7226ce554f461b"  # 替换为您在 tushare 中申请的 token
ts.set_token(token)
pro = ts.pro_api()

# 交易日历
TODAY = datetime.datetime.now().strftime("%Y%m%d")
tradedays = load_object_from_pickle("tradedays")


def get_tradedays(start_date="20060101", end_date=TODAY, first_day_of_this_month=False):
    if first_day_of_this_month:
        start_date = datetime.datetime.today().replace(day=1).strftime("%Y-%m-%d")
    tradedays = pro.query("trade_cal", start_date=start_date, end_date=end_date)
    tradedays = pd.to_datetime(tradedays[tradedays["is_open"] == 1]["cal_date"])
    tradedays = tradedays.sort_values(ascending=True).to_list()
    return tradedays


def get_tradeassets_for_stock():
    # 从股票行情数据中取出stock的symbol
    # 从指数的行情数据取出benchmark的symbol
    basic_info = load_object_from_pickle("basic_info_stock")
    tradeassets = load_object_from_pickle("stock_close")
    tradeassets = tradeassets.columns.tolist()
    tradeassets = [
        i
        for i in tradeassets
        if i in basic_info["symbol"].to_list()
           and i not in ["301329", "301456", "301503", "833751"]
    ]

    return tradeassets


def get_tradeassets_for_index(index_estclass: IndexEstclass):
    # 从basic info直接取出symbol
    basic_info = get_basic_info_for_index(index_estclass)
    tradeassets = list(basic_info["symbol"].unique())
    # tradeassets = [i for i in tradeassets if
    #                i in basic_info['symbol'].to_list() and i not in ['301329', '301456', '301503', '833751']]
    return tradeassets


def get_trade_calendar_for_stock():
    """
    Define the tradeable stock pool
        - 1. List status == 'L'
        - 2. Is ST == 'N'
        - 3. Is delisted == 'N'

    """

    basic_info = load_object_from_pickle("basic_info_stock").query(
        "liststatus=='1' or liststatus=='0'"
    )
    tradeassets = load_object_from_pickle("tradeassets_stock")

    # 上市中
    liststatus_df = pd.DataFrame(index=tradedays, columns=tradeassets)
    basic_info["listdate"] = pd.to_datetime(basic_info["listdate"])
    basic_info["delistdate"] = pd.to_datetime(basic_info["delistdate"])

    for row in basic_info.itertuples():
        symbol = row.symbol
        listdate = row.listdate if row.listdate.year != 1900 else None
        delistdate = row.delistdate if row.delistdate.year != 1900 else None
        if symbol not in tradeassets:
            continue
        if listdate and delistdate:
            # Update the corresponding value in liststatus_df
            liststatus_df.loc[
                (liststatus_df.index >= listdate) & (liststatus_df.index <= delistdate),
                symbol,
            ] = 1
        if listdate and not delistdate:
            liststatus_df.loc[(liststatus_df.index >= listdate), symbol] = 1

    # 非次新股
    listday_df = pd.DataFrame(index=tradedays, columns=tradeassets)
    listday_df[liststatus_df.cumsum() > 120] = 1

    # ～ST
    st_df = pd.DataFrame(index=tradedays, columns=tradeassets)
    st_stocks = list(basic_info.query('sesname.str.contains("ST")').symbol)
    st_df.loc[:, st_df.columns.isin(st_stocks)] = 1
    not_st_df = st_df.isna().astype(bool)

    # ～停牌
    volume = load_object_from_pickle("stock_volume")
    suspend_df = volume.isna().astype(bool)
    not_suspend_df = ~suspend_df

    # 成交额大于5000w
    # amount = load_object_from_pickle("stock_amount")
    # amount_df = (amount > 20000000).astype(bool)

    # ～当日涨停
    close = load_object_from_pickle("stock_close")
    kechuang_board = basic_info.query('board=="4"').symbol.tolist()
    kechuang_board = [x for x in kechuang_board if x in tradeassets]
    pchg = close.pct_change()
    lift_limit_df = pd.DataFrame(index=tradedays, columns=tradeassets)

    lift_limit_df.loc[:, kechuang_board] = (pchg.loc[:, kechuang_board] > 0.195).astype(
        int
    )
    lift_limit_df.loc[:, ~lift_limit_df.columns.isin(kechuang_board)] = (
            pchg.loc[:, ~pchg.columns.isin(kechuang_board)] > 0.095
    ).astype(int)
    lift_limit_df = lift_limit_df.astype(bool)
    not_lift_limit_df = ~lift_limit_df

    # ～负净资产
    # righaggr = load_object_from_pickle('stock_righaggr')
    # righaggr = righaggr.loc[righaggr.index.isin(tradedays),righaggr.columns.isin(tradeassets)].ffill()
    # neg_asset_df = (righaggr<0).astype(int)
    # neg_asset_df = neg_asset_df.astype(bool)
    # not_neg_asset_df = ~neg_asset_df

    trade_calendar = (
            liststatus_df * listday_df * not_st_df * not_suspend_df * not_lift_limit_df 
    )
    trade_calendar = trade_calendar.replace({0: np.nan})
    trade_calendar = trade_calendar.loc[tradedays, tradeassets].ffill()  # in case of null value

    return trade_calendar


def get_stock_pool_from_scope(scope="all"):
    tradeassets = load_object_from_pickle("tradeassets_stock")
    trade_calendar = load_object_from_pickle("trade_calendar")
    stock_pool = pd.DataFrame(index=tradedays, columns=tradeassets)

    if scope == "all":
        stock_pool.loc[:, :] = 1
    else:
        index_components = load_object_from_pickle('index_components')
        index_components.columns = index_components.columns.str.lower()
        df = index_components.query("symbol==@scope")
        df["selecteddate"] = pd.to_datetime(df["selecteddate"])
        df["outdate"] = pd.to_datetime(df["outdate"])

        for row in df.itertuples():
            symbol = row.selectedsymbol
            selecteddate = row.selecteddate if row.selecteddate.year != 1900 else None
            outdate = row.outdate if row.outdate.year != 1900 else None
            if selecteddate and outdate:
                stock_pool.loc[
                    (stock_pool.index >= selecteddate) & (stock_pool.index < outdate),
                    symbol,
                ] = 1
            if selecteddate and not outdate:
                stock_pool.loc[(stock_pool.index >= selecteddate), symbol] = 1
    stock_pool = stock_pool * trade_calendar

    return stock_pool