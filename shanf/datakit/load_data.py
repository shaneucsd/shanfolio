import logging
import numpy as np
import pandas as pd 
import os
import datetime
import tushare as ts
from WindPy import w

DATA_PATH = '../source/stock'
OUTPUT_PATH = '../processed/stock'
LOGGING_PATH = '../log/dataloader.log'
START_DATE = '2009-01-01'
END_DATE = '2022-06-30'
EXCHANGE = 'ASHARE'

KWARGS_DICT ={
    'sk_daily': {'parse_dates':['tradedate']},
    'shhk': {'parse_dates':['tradedate']},
    'moneyflow': {'parse_dates':['tradedate']},
    'forecast': {'parse_dates':['tradedate']},
    'valuation':{'parse_dates':['tradedate']},
    'financials':{'parse_dates':['declaredate']},
}

logging.basicConfig(filename=LOGGING_PATH,level=logging.INFO, datefmt='## %Y-%m-%d %H:%M')

TODAY = datetime.datetime.now().strftime("%Y%m%d")
token = "4de969c81ef5f589fed566bb5e3e89808d86c59eed7226ce554f461b"  # 替换为您在 tushare 中申请的 token
ts.set_token(token)
pro = ts.pro_api()


class SourcePiper():

    """
    Default reading Method is Parquet

    Data Include:
        - SK data daily
        
    """

    global DATA_PATH, OUTPUT_PATH, START_DATE, END_DATE, EXCHANGE
    pathmap = {}

    def __init__(self,ori_root, save_root, level_0='sk_daily',insert=True) -> None:
        self.ori_root = ori_root if ori_root else DATA_PATH
        self.save_root = save_root if save_root else OUTPUT_PATH

        self.path = os.path.join(self.ori_root,level_0)
        self.level_0 = level_0
        self.insert = insert

        self._assert_path(self.path)
        self._map_path()

        logging.info(f'Initialized piper in {self.path}')


    def _map_path(self):
        self.pathmap = {}
        self.pathmap.update({self.__name_modify(name): (self.path,name) for name in os.listdir(self.path)})
        special_list = []
        for name, items in self.pathmap.items():
            if 'curr' in name or 'past' in name:
                special_list.append(name[:-5])

        for name in set(special_list):
            self.pathmap.update({name: [(self.path,f'{name}_curr.csv'),(self.path,f'{name}_past.csv')]})
            self.pathmap.pop(f'{name}_curr')
            self.pathmap.pop(f'{name}_past')

    def _assert_path(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
            logging.info(f'No directory in {path}')
            raise FileNotFoundError(f'No directory in {path}')

    def _get_kwargs(self):
        if self.level_0 in KWARGS_DICT.keys():
            return KWARGS_DICT[self.level_0]
        else:
            raise KeyError(f'No kwargs for {self.level_0}')
    
    def __name_modify(self,f_name):
        if f_name.endswith('.csv'):
            f_name = f_name.split('.')[0]
        return f_name.lower()
    
    # Read Single File
    def _read_file(self,path,filename,**kwargs):
        
        ext = filename.split('.')[-1]

        if ext == 'csv':
            df = pd.read_csv(os.path.join(path,filename),encoding = 'gbk')
            df = self._format(df, **kwargs)
            msg = f'Retrieved df, shape of {df.shape}, of {os.path.join(path,filename)}, modified at {get_file_modify_time(os.path.join(path,filename))}'
            logging.info(msg)

        else:
            msg = f'Unsupported file type: {ext}, of {os.path.join(path,filename)}'
            logging.info(msg)
            raise TypeError(msg)
        return df 
    
    def _read_multiple_file(self,items,**kwargs):
        def _yield_df(items):
            for i in range(len(items)):
                path, filename = items[i][0], items[i][1]
                df = self._read_file(path,filename,**kwargs)
                yield df
            
        dfs = [df for df in _yield_df(items)]
        assert len(dfs)>0, 'No Data in Directory, {items}'
        df = pd.concat(dfs)
        return df


    def _format(self, df, **kwargs):
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
            df.drop('Unnamed: 0',axis=1,inplace=True)

        # Format Date
        for col in kwargs['parse_dates']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col],format='%Y%m%d')
        
        # Remove Duplicated Rows
        df.loc[:,:] = df[~df.duplicated()]
        df = df[df['symbol'].notna()]
        
        # Set Tradedate as index
        df.reset_index(inplace=True)        
        
        if 'index' in df.columns:
            df.drop('index',axis=1,inplace=True)
        
        # Format Symbol
        if EXCHANGE == 'ASHARE':
            df['symbol'] = df['symbol'].apply(lambda x: convert_symbol(x))
        return df    



    def _pivot_and_save(self, df):

        def _pivot_col(df,col):
            pivot_df = df.pivot(index='tradedate',columns='symbol',values=col)
            return pivot_df

        def _save(df,file_path):
            # 检查是否存在本地parquet文件
            if os.path.exists(file_path) and self.insert:
                # 读取本地parquet文件，插入新数据
                existing_df = pd.read_pickle(file_path)        
                df = pd.concat([existing_df, df])
                df = df[~df.index.duplicated(keep='last')]
                df.to_pickle(path)
            else:
                df = df[~df.index.duplicated(keep='last')]
                df.to_pickle(path)

            return df 
        
        print('Data From: ',df.tradedate.min(),' to ',df.tradedate.max())

        target_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64','int64'] and col not in ['tradedate','symbol','secode','index']:
                target_cols.append(col)

        for col in target_cols:
            path = os.path.join(self.save_root,f'{col}.pkl')
            pivot_df = _pivot_col(df,col)
            _save(pivot_df,path)
            
        msg = f"Insert: {self.insert}, Data from {pivot_df.index[0]} to {pivot_df.index[-1]} has been saved to {self.save_root}, columns: {target_cols}"
        print(msg)
        
    def _insert(self,items, **kwargs):
        for path,filename in items:
            if 'curr' in filename:
                df = self._read_file(path,filename,**kwargs)
                return df

    # Pivot Factor data to matrix
    def _transform(self):
        
        kwargs = self._get_kwargs()
        
        for f_name, items in self.pathmap.items():
            
            logging.info(f'Reading {f_name}...')

            if isinstance(items,tuple):
                path,filename = items
                df = self._read_file(path,filename,**kwargs)

            elif isinstance(items,list):
                if self.insert:
                    df = self._insert(items,**kwargs)
                else:
                    df = self._read_multiple_file(items,**kwargs)

            else:
                raise TypeError(f'Unsupported type: {type(items)}')
            
            self._pivot_and_save(df)
            logging.info(f'Finished {f_name}...')



class FactorLoader():
    
    def __init__(self,factor_root, tradedays) -> None:
        self.factor_root = factor_root
        self.factor_dict = {}
        self._map_path(self.factor_root)
        self.tradedays = tradedays

    def _set_default_days_assets(self, days=None, assets=None):
        self._days = days
        self._assets = assets
        self._screen_days = False if self._days is None else True
        self._screen_assets = False if self._assets is None else True

    def _map_path(self, path):
        self.pathmap = {}
        self.pathmap.update({f_name: (path, name) for f_name,name in self._yield_path(path)})

    def _yield_path(self, path):
        
        for name in os.listdir(path):
            if name.endswith('.pkl'):
                f_name = name.split('.')[0].lower()
                yield f_name, name
            else:
                pass

    def _read_file(self, path, name):
        df = pd.read_pickle(os.path.join(path, name))
        df.index = pd.to_datetime(df.index)
        if self.tradedays:
            try:
                df = df.loc[self.tradedays]
            except:
                pass
        return df

    def __getattr__(self,f_name):

        if f_name in self.__dict__.keys():
            return self.__dict__[f_name]
        
        # init a new object if f_name is in ['_days','_assets','_screen_days','_screen_assets']
        if f_name.startswith('_'): 
            self.__dict__[f_name] = None
            return self.__dict__[f_name]

        if f_name not in self.pathmap.keys():
            try:
                f_name = self._get_factor_valid_name(f_name)
            except:
                raise ValueError('Factor name is not in the factor list')
        
        if f_name not in self.factor_dict.keys():
            path, name = self.pathmap[f_name]
            self.factor_dict[f_name] = self._read_file(path, name)

        if self._screen_assets:
            self.factor_dict[f_name] = self.factor_dict[f_name].loc[:, self.factor_dict[f_name].columns.isin(self._assets)]
        if self._screen_days:
            self.factor_dict[f_name] = self.factor_dict[f_name].loc[ self.factor_dict[f_name].index.isin(self._days),:]
            
        return self.factor_dict[f_name]

    def _get_factor_valid_name(self,f_name):
        # try to map the f_name with the existing factor name. If f_name is the start one of the factor name in the factor_dict, then return the factor name. if not , return None
        factor_name = [ factor_name for factor_name in self.pathmap.keys() if factor_name.startswith(f_name)]
        if len(factor_name) == 1:
            return factor_name[0]
        if len(factor_name) > 1:
            raise ValueError(f'Factor name {f_name} is not unique, follows by {factor_name}')
        else:
            raise ValueError(f'Factor name {f_name} is not in the factor list')

class WindIndexAppender():

    wind_index_dict = {
        '8841415.WI':'茅指数', # 茅台指数
        '8841447.WI':'宁指数', # 宁德时代
        '8841660.WI':'AIGC', # AIGC
        'CI009187.WI':'复合集流体', # 复合集流体,
        '8841681.WI':'中特估',
        '8841422.WI':'顺周期',
        '8841241.WI':'存储器指数',
        '8841678.WI':'算力指数',
        '884201.WI':'人工智能',
        '8841418.WI':'医美指数',
        
    }
    
    curr_df_path = '../source/index/sk_daily/index_market_curr.csv'
    past_df_path = '../source/index/sk_daily/index_market_past.csv'
    index_comp_path = '../source/index/index_comp/index_comp.csv'

    def __init__(self):
    
        w.start()
        self.tradeday = self.get_last_tradeday()
        print('wind_index_dict',self.wind_index_dict)

    def get_last_tradeday(self):
        # get today datetime
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        # get the first day of this month
        first_day_of_this_month = datetime.datetime.today().replace(day=1).strftime('%Y-%m-%d')

        e,tradedays = w.tdays(first_day_of_this_month,today, "",usedf=True)
        tradeday = tradedays.index[0].strftime('%Y-%m-%d')
        return tradeday

    def get_sk_daily_df_from_wind(self,symbol, start, end):
        e, wind_sk_daily_df = w.wsd(symbol, "close,low,high,open,volume,amt,pct_chg,val_mvc,val_mv_ARD", start, end, "unit=1;PriceAdj=B",usedf=True)
        wind_sk_daily_df = self.format_wind_sk_daily_df(symbol, wind_sk_daily_df)

        return wind_sk_daily_df

    def get_index_comp_from_wind(self, symbol):
        e, wind_index_comp_df = w.wset("sectorconstituent",f"date={self.tradeday};windcode={symbol}",usedf=True)
        wind_index_comp_df = self.format_wind_index_comp_df(symbol, wind_index_comp_df)
        return wind_index_comp_df
        
    def format_wind_index_comp_df(self,symbol, wind_index_comp_df):
        
        wind_index_comp_df = wind_index_comp_df.rename(columns = {'wind_code':'selectesymbol','sec_name':'selectesesname'})
        wind_index_comp_df['secode']= np.nan
        wind_index_comp_df['symbol'] = symbol
        wind_index_comp_df['sesname'] = self.wind_index_dict[symbol]
        wind_index_comp_df['selectesymbol'] = wind_index_comp_df['selectesymbol'].apply(lambda x: x[:6])
        if 'date' in wind_index_comp_df.columns:
            wind_index_comp_df = wind_index_comp_df.drop(columns='date')
        wind_index_comp_df = wind_index_comp_df [['secode', 'symbol', 'selectesymbol', 'selectesesname', 'sesname']]
        
        return wind_index_comp_df

    def format_wind_sk_daily_df(self,symbol,wind_sk_daily_df):
        
        wind_sk_daily_df = wind_sk_daily_df.reset_index()
        wind_sk_daily_df['SYMBOL'] = symbol
        wind_sk_daily_df['ESTCLASS'] = '中证主题指数'
        wind_sk_daily_df['SESNAME'] =  self.wind_index_dict[symbol]
        wind_sk_daily_df['SECODE'] = None
        wind_sk_daily_df['ENTRYDATE'] = None
        wind_sk_daily_df['ENTRYTIME'] = None

        wind_sk_daily_df = wind_sk_daily_df.rename(columns={'index':'TRADEDATE','AMT':'AMOUNT','PCT_CHG':'PCHG','VAL_MVC':'NEGOTIABLEMV','VAL_MV_ARD':'TOTMKTCAP'})
        wind_sk_daily_df = wind_sk_daily_df[['TRADEDATE', 'SECODE', 'SYMBOL', 'ESTCLASS', 'SESNAME', 'OPEN', 'CLOSE',
            'HIGH', 'LOW', 'VOLUME', 'AMOUNT', 'PCHG', 'NEGOTIABLEMV', 'TOTMKTCAP',
            'ENTRYDATE', 'ENTRYTIME']]
        
        wind_sk_daily_df['TRADEDATE'] = wind_sk_daily_df['TRADEDATE'].apply(lambda x: x.strftime('%Y%m%d'))
        
        return wind_sk_daily_df

    def read_local_index_comp_df(self):
        def convert_symbol(x):
        # if x can be converted to int
            try:
                x = str(int(x)).zfill(6)
            except:
                pass
            return x

        # Index Compontent
        index_comp_df = pd.read_csv(self.index_comp_path,encoding='gbk')
        index_comp_df.columns = index_comp_df.columns.str.lower()
        index_comp_df['symbol'] = index_comp_df['symbol'].apply(lambda x: convert_symbol(x))
        index_comp_df['selectesymbol'] = index_comp_df['selectesymbol'].apply(lambda x: convert_symbol(x))

        return index_comp_df

    def append_to_local_index_comp_df(self):

        index_comp_df = self.read_local_index_comp_df()

        for symbol in self.wind_index_dict.keys():
            wind_index_comp_df = self.get_index_comp_from_wind(symbol)
            len_1 = len(wind_index_comp_df)
            index_comp_df = index_comp_df.append(wind_index_comp_df)
            index_comp_df.drop_duplicates(inplace=True)

        index_comp_df.to_csv(self.index_comp_path,encoding='gbk',index=False)

    def append_to_local_sk_daily_df(self,path):
        print(path)
        df = pd.read_csv(path,encoding='gbk')
        tradedays = pd.Series(pd.to_datetime(df['TRADEDATE'],format='%Y%m%d').sort_values().unique()).to_list()

        for symbol in self.wind_index_dict.keys():
            curr_len = len(df)
            wind_sk_daily_df = self.get_sk_daily_df_from_wind(symbol, tradedays[0],tradedays[-1])
            df = df.append(wind_sk_daily_df)
            df.drop_duplicates(inplace=True)
        df.to_csv(path,encoding='gbk',index=False)
        return df


class GeneralLoader():

    def __init__(self,ori_root=None,save_root=None):
        self.ori_root = ori_root
        self.save_root = save_root

    def _get_index_comp(self,path='../source/index/index_comp/index_comp.csv'):

        sec_index_comp_df = pd.read_csv(path,encoding='gbk')
        # lower all columns 
        sec_index_comp_df.columns = sec_index_comp_df.columns.str.lower()
        # fill symbol with 0
        sec_index_comp_df['symbol'] = sec_index_comp_df['symbol'].apply(lambda x: str(x).zfill(6))
        sec_index_comp_df['selectesymbol'] = sec_index_comp_df['selectesymbol'].apply(lambda x: str(x).zfill(6))

        sec_index_comp_dict = {}
        for symbol, group in sec_index_comp_df.groupby('symbol')['selectesymbol']:
            sec_index_comp_dict.update({symbol: group.tolist()})
            assert len(group.tolist()) > 0, 'No data for {}'.format(symbol)

        return sec_index_comp_dict
    
    def _get_risk_free_rate(self,path='../source/macro/国债到期收益率_10年.xlsx'):
        risk_free_rate = pd.read_excel(path,parse_dates=True,index_col=[0])[4:]
        risk_free_rate.index = pd.to_datetime(risk_free_rate.index)
        risk_free_rate.index.name = 'tradedate'
        risk_free_rate = (1+risk_free_rate[['国债到期收益率:10年']]/100) **(1/252)-1 # 转化为日收益率
        return risk_free_rate
    
    

class CombinedFactorDf():
    '''
    Create a abstract class for asset combined factor df
    Attributes:
        - asset_type : 'index' or 'stock'
        - asset_type_detailed : '行业指数' or '宽基指数' or '主题指数'
        - df : 
            {
            'tradedate' : ['2019-01-01','2019-01-02',...],
            'symbol' : ['000001','000002',...],
            'factor_1' : [1,2,...],
            'factor_2' : [1,2,...],
            }
        - factor_name : ['factor_1','factor_2',...]
    '''
    df = None
    factor_names = []

    def __init__(self,
                 asset_type:str,
                 asset_type_detailed:str,
                 tradedays,
                 tradeassets,
                 ):
        self.asset_type = asset_type
        self.asset_type_detailed = asset_type_detailed
        self.tradedays = tradedays
        self.tradeassets = tradeassets

        self.init_df(tradedays,tradeassets)
    
    # init dataframe with tradedays and tradeassets
    def init_df(self,tradedays,tradeassets):
        multi_index = pd.MultiIndex.from_product([tradedays, tradeassets], names=['tradedate', 'symbol'])
        self.df = pd.DataFrame(index=multi_index, columns=[])
        
    # reset the df
    def reset_df(self):
        self.df = None
        self.factor_names = []
        self.init_df(self.tradedays,self.tradeassets)

    def __repr__(self):
        return f'CombinedFactorDf(asset_type={self.asset_type},asset_type_detailed={self.asset_type_detailed})'
    
    # insert a factor into df by merging
    def insert_factor(self, factor_df, name='default'):        

        if not self.df.index.levshape == factor_df.shape:
            factor_df = factor_df.loc[self.tradedays,self.tradeassets]
        
        # Format the factor_df
        factor_df.index.name = 'tradedate'
        factor_df.columns.name = 'symbol'
        factor_df = pd.DataFrame(factor_df.unstack(),columns=[name])

        if self.df is None:
            self.df = self.init_df(self.tradedays,self.tradeassets)

        # check if the factor name is already in the df
        if name in self.factor_names:
            # if yes, print warning
            print(f'Warning: {name} is already in the df, will be replaced')
            # drop the factor
            self.df = self.df.drop(name,axis=1)

        self.df = pd.merge(self.df,factor_df,on=['tradedate','symbol'],how='outer')

        self.factor_names = list(self.df.columns)



class Exporter():
    '''
        Create a class for formatting the combined factor df, and export to sql
        Adding some formatted methods
        Distinct Attributes:
            - info_df:
            {
            'symbol' : ['000001','000002',...],
            'sesname' : ['农林牧渔','食品饮料',...],
            'estclass' : ['申万一级行业指数','申万一级行业指数',...],
            'indexname': ['申万农林牧渔指数','申万食品饮料指数',...],
            }
        
    '''
    # init the dataframe with info_df

    
    def __init__(self,
                 info_df:pd.DataFrame=None,
                 df:pd.DataFrame = None,
                 factor_names:list = None,
                 asset_type:str = None):
        self.info_df = info_df
        self._df = df
        self.factor_names = factor_names
        self.asset_type = asset_type
    
    
    @property
    def df(self):
        # Reset the index to make sure the index is not multi-index
        # check if the index is multi-index
        if isinstance(self._df.index,pd.MultiIndex) and (self._df.index.names[0],self._df.index.names[1]) == ('tradedate','symbol'):
            self._df = self._df.reset_index()
        return self._df

    @property
    def sql_df(self):
        df = self.merge_df_with_info(self.df)
        df = self.factor_to_numerical(df)
        df = df.set_index(['tradedate','symbol'])
        return df
    
    def factor_to_numerical(self,df):
        """
        DataEase 不支持字段名称的修改，否则现有仪表板会报错，所以需要将因子名称转换为数字
        预留100个字段，以防后续因子增加
        """
        df = df.rename(columns=dict(zip(self.factor_names,range(len(self.factor_names)))))
        # 预留100个字段，以防后续因子增加
        for i in range(len(self.factor_names),100):
            df[i] = np.nan
        return df
    
    def merge_df_with_info(self,df):
        df = pd.merge(df,self.info_df,on='symbol',how='left')
        if self.asset_type == 'index':
            df = df[['tradedate','symbol','sesname','estclass','indexname']+self.factor_names]
        return df
    
    def export_to_sql(self,table_name:str,engine, if_exists='append'):

        sql_df = self.sql_df

        if if_exists == 'replace':

            pass

        else:


            try:
                # In order to save time, will first retrieve existing data from sql, and then merge with the new data
                # to avoid the duplicated data
                records_query = f"select symbol, tradedate from broad_index"
                records = pd.read_sql(records_query, engine, index_col=['tradedate', 'symbol'])
                sql_df = sql_df[~sql_df.index.isin(records.index)]
                print(f'    Append: {len(sql_df)}, Exisiting: {len(records)} in {table_name} ')
                
            except:
                records = []
                if_exists = 'replace'
                print(f'No data in {table_name}, will replace the table')
                
        sql_df = sql_df.reset_index()
        sql_df.to_sql(table_name,engine,if_exists=if_exists,index=False)

        print(f'Type: {if_exists}, Insert {len(sql_df)} records into {table_name}')
        
        

def get_tradedays(start_date='20060101', end_date=TODAY, first_day_of_this_month=False):
    if first_day_of_this_month:
        start_date = datetime.datetime.today ().replace(day=1).strftime('%Y-%m-%d')
    tradedays = pro.query('trade_cal', start_date=start_date, end_date=end_date)
    tradedays = pd.to_datetime(tradedays[tradedays["is_open"] == 1]["cal_date"])
    tradedays = tradedays.sort_values(ascending=True).to_list()
    return tradedays

def get_tradeassets():
    tradeassets = pro.stock_basic(exchange='', list_status='L', fields='symbol')
    tradeassets = tradeassets["symbol"].to_list()
    return tradeassets

def get_listdate(matrix=False,tradedays=None):
    listdate = pro.stock_basic(exchange='', list_status='L', fields='symbol,list_date')
    listdate['list_date'] = pd.to_datetime(listdate['list_date'])
    listdate = listdate.set_index("symbol").to_dict()["list_date"]
    if matrix:
        listdate = pd.DataFrame(get_listdate(),index=tradedays)
        listdate = listdate.apply(lambda x: np.where(x.index > x , 1, 0)).cumsum()
    return listdate

def pivot_col(df,col):
    df = pd.pivot(df,columns='symbol',values=col)
    df = df.replace(0,np.nan)
    df.ffill(inplace=True)
    return df

def dict_to_df(_dict):
    df = pd.DataFrame(_dict,index=['sesname']).T
    df.index.name='symbol'
    df.reset_index(inplace=True)
    return df

# check the modified time of a file
def get_file_modify_time(file_path):
    t = os.path.getmtime(file_path)
    # format to string 
    s = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    return s

def get_index_info_comp_df_from_local(info_path,comp_path):
    def convert_symbol(x):
    # if x can be converted to int
        try:
            x = str(int(x)).zfill(6)
        except:
            pass
        return x
    index_info_df = pd.read_csv(info_path,encoding='gbk')
    index_info_df.columns = index_info_df.columns.str.lower()
    index_info_df['symbol'] = index_info_df['symbol'].apply(lambda x: convert_symbol(x))
    
    index_comp_df = pd.read_csv(comp_path,encoding='gbk')

    index_comp_df.columns = index_comp_df.columns.str.lower()
    index_comp_df['symbol'] = index_comp_df['symbol'].apply(lambda x: convert_symbol(x))
    index_comp_df['selectesymbol'] = index_comp_df['selectesymbol'].apply(lambda x: convert_symbol(x))

    return index_info_df,index_comp_df