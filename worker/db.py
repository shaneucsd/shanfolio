import sys

sys.path.append("../dags")
from postprocess.datakit.DtLoader import DtLoader
from postprocess.datakit.DtTransform import DtTransform
from utils.file_utils import (
    PROCESSED_FACTOR_PATH,
    save_object_to_pickle_with_path,
    load_object_from_pickle_with_path,
    load_object_from_pickle,
)
import numpy as np
import pandas as pd
from utils.operator_utils import *
from enums.index_estclass import IndexEstclass
from utils.backtest_utils import *
from funcs import *

directory = PROCESSED_FACTOR_PATH + "/stock"
dl = DtLoader("stock")
dt = DtTransform("stock")



def concat_factor(factors: dict, directory:str, start=None, end=None):
    slice_x = slice(start, end)
    factor_df_list = []
    names = []
    for key, factor_list in factors.items():
        _dir = directory + "/" + key 
        for factor in factor_list:
            _factor = load_object_from_pickle_with_path(factor, _dir).loc[slice_x, :]
            factor_df_list.append(_factor)
            names.append(factor)
            
    values_list = []
    tradedates = factor_df_list[0].index.values
    symbols = factor_df_list[0].columns.values

    tradedates_long = np.tile(tradedates, len(symbols))
    symbols_long = np.repeat(symbols, len(tradedates))

    for i,df in enumerate(factor_df_list):
        values = df.values.flatten(order='F')
        values_list.append(values)

    values_concat = np.column_stack(values_list)

    data = pd.DataFrame({
        'tradedate': tradedates_long,
        'symbol': symbols_long,
    })

    for i, col_name in enumerate(names):
        data[col_name] = values_concat[:, i]
        
    return data