from enums.index_estclass import IndexEstclass
from utils.file_utils import read_pickle_from_sql_tasks


def get_basic_info_for_stock():
    basic_info = read_pickle_from_sql_tasks(f'info_stock_basic_info')
    basic_info['symbol'] = basic_info['symbol'].astype(str).apply(lambda x: x.zfill(6))
    basic_info = basic_info.drop_duplicates(subset=['symbol'], keep='first')
    return basic_info


def get_basic_info_for_index(index_estclass: IndexEstclass):
    basic_info = read_pickle_from_sql_tasks(f'info_index_basic_info_{index_estclass.value}')
    basic_info['symbol'] = basic_info['symbol'].astype(str).apply(lambda x: x.zfill(6))
    basic_info = basic_info.drop_duplicates(subset=['symbol'], keep='first')
    return basic_info


def get_symbol_sesname_dict_for_stock():
    basic_info_stock = get_basic_info_for_stock()
    basic_info_stock_dict = dict(zip(basic_info_stock['symbol'], basic_info_stock['sesname']))
    return basic_info_stock_dict
