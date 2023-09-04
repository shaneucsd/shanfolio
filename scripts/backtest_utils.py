import os 
import pandas as pd
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('../config.ini')
data_path = config.get('path', 'data_path')


def get_benchmark_close(symbol = '000300'):  
    bm_close = pd.read_pickle(os.path.join(data_path,'processed/index/close.pkl'))[symbol].ffill()
    return bm_close

def get_benchmark_return(symbol = '000300'):
    bm_close = get_benchmark_close(symbol)
    bm_ret = np.log(bm_close).diff()
    return bm_ret


