import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
from enums.asset_type import AssetType
from enums.index_estclass import IndexEstclass
from postprocess.datakit.DtSource import FACTOR_PATH
from utils.basic_info import get_basic_info_for_index
from utils.file_utils import load_object_from_pickle
from utils.trade_date_assets import get_tradeassets_for_index


class DtLoader:
    """
    DtLoader
    
    
    Read fields    
    
    """

    def __init__(self, prefix, index_estclass: IndexEstclass = None) -> None:
        """
        prefix : 'stock' or 'index'
        asset_type : AssetType.STOCK, AssetType.INDEX

        @field_dict: dict, {field_name: df}, stores all the fields
        @field_root: str, the root path of the fields
        @pathmap: dict, {field_name: (path, file_name)}

        """

        self.prefix = prefix
        self.asset_type = AssetType.STOCK if prefix == 'stock' else (AssetType.INDEX if prefix == 'index' else None)

        # 这里Load的时候，要根据asset type，Load不同的Tradeassets和Tradedays,basic_info同理
        if index_estclass is not None:  # 只有index走这里
            self.basic_info = get_basic_info_for_index(index_estclass)
            self.tradeassets = get_tradeassets_for_index(index_estclass)
        else:
            self.basic_info = load_object_from_pickle(f'basic_info_stock')
            self.tradeassets = load_object_from_pickle(f'tradeassets_stock')
        self.tradedays = load_object_from_pickle('tradedays')

        # 每种资产独有的properties
        if prefix == 'stock':
            # 股票池
            # all_stock_pool = load_object_from_pickle('all_stock_pool') # 全市场
            # csiall_stock_pool = load_object_from_pickle('csiall_stock_pool') # 中证全指
            # hs300_stock_pool = load_object_from_pickle('hs300_stock_pool') # 沪深300
            # csi500_stock_pool = load_object_from_pickle('csi300_stock_pool') # 中证500
            # csi800_stock_pool = load_object_from_pickle('csi800_stock_pool') # 中证800
            # csi1000_stock_pool = load_object_from_pickle('csi1000_stock_pool') # 中证1000
            # csi2000_stock_pool = load_object_from_pickle('csi2000_stock_pool') # 中证2000
            # 行业暴露
            pass

        if prefix == 'index':
            # Read by Index Type as well as Index Components.
            # TODO: 读取Index Components 并制作成一个Components矩阵
            #                symbol symbol ....
            #

            self.index_comp_df = load_object_from_pickle('index_components')
            self.index_comp_df['selecteddate'] = pd.to_datetime(self.index_comp_df['selecteddate'], format='%Y%m%d')
            self.index_comp_df['outdate'] = pd.to_datetime(self.index_comp_df['outdate'], format='%Y%m%d')
            # 获取今日数据
            today = datetime.today()
            self.index_comp_df = self.index_comp_df[
                (self.index_comp_df['selecteddate'] <= today) &
                (self.index_comp_df['outdate'] >= today) &
                (self.index_comp_df['symbol'].isin(self.tradeassets))
                ]
            index_comp_dict = {}
            for symbol, group in self.index_comp_df.groupby('symbol')['selectedsymbol']:
                if symbol in self.tradeassets:
                    index_comp_dict.update({symbol: group.tolist()})
            self.index_comp_dict = index_comp_dict

        self.field_dict = {}
        self.field_root = os.path.join(FACTOR_PATH, f'{prefix}')
        self.pathmap = self._map_path(self.field_root)  # yield a dict for fields

    def _map_path(self, path):
        return {f_name: (path, name) for f_name, name in self._yield_path(path)}

    def _yield_path(self, path):
        path_obj = Path(path)
        for file_path in path_obj.iterdir():
            if file_path.suffix == '.pkl':
                f_name = file_path.stem.lower()
                yield f_name, file_path.name

    def _read_file(self, path, name):
        file_path = Path(path) / name
        df = pd.read_pickle(file_path)
        df.index = pd.to_datetime(df.index)
        return df

    def _align_trade_date_asset_for_normal_field(self, df):
        # Align tradedays and index Meanwhile
        # Should be Finished in DtSource, when pivoting.
        df = df.reindex(columns=self.tradeassets)
        df = df.reindex(index=df.index.union(self.tradedays))
        df = df.loc[self.tradedays]
        return df

    def _align_trade_date_asset_for_financial_field(self, df):
        # Align Trade assets
        # Financial Values were left null if 0.
        # Replace 0 first
        # Align Trade days without losing data, union index, ffill

        df = df.reindex(columns=self.tradeassets)
        df = df.reindex(df.index.union(self.tradedays)).ffill()
        df = df.loc[self.tradedays]
        return df

    def get_field_df(self, field):
        path, file_name = self.pathmap[field]
        df = self._read_file(path, file_name)
        df = self._align_trade_date_asset_for_normal_field(df)
        return df

    def get_financial_field_df(self, field):
        path, file_name = self.pathmap[field]
        df = self._read_file(path, file_name)
        df = self._align_trade_date_asset_for_financial_field(df)
        df = df.fillna(0)
        return df

    def reload(self):
        to_pop = []
        for field in self.__dict__.keys():
            if field in self.pathmap.keys():
                to_pop.append(field)
        for field in to_pop:
            self.__dict__.pop(field)

    def cached_field(field_func):
        def wrapper(self, field):
            if field not in self.__dict__.keys():
                df = self.get_field_df(field)
                self.__dict__[field] = df
            return field_func(self, field)

        return wrapper

    # TODO: slc: 加注释，怎么使用
    @cached_field
    def __getattr__(self, field):
        return self.get_field_df(field)

    
        
    def get_stk_industry(self) -> pd.DataFrame:
        # 行业: 截止日的行业分类 // 中信行业
        def datetime_to_int(dt):
            try:
                return int(dt.strftime('%Y%m%d'))
            except:
                return int(dt)
        end_dt = int(self.tradedays[-1].strftime('%Y%m%d'))
        # Index Components
        index_components  = load_object_from_pickle(tag='index_components')
        index_components.outdate = index_components.outdate.map(datetime_to_int)
        swlevel1_index_components = index_components.query('(estclass=="申万一级行业指数") & (usestatus=="1") & outdate > @end_dt')
        swlevel1_dict = dict(zip(swlevel1_index_components['selectedsymbol'],swlevel1_index_components['symbol']))

        swlevel3_index_components = index_components.query('(estclass=="申万三级行业指数") & (usestatus=="1") & outdate > @end_dt')
        swlevel3_dict = dict(zip(swlevel3_index_components['selectedsymbol'],swlevel3_index_components['symbol']))
    
        # 1. 创建一个新的DataFrame，索引为self.tradeassets
        industry_df = pd.DataFrame(index=self.tradeassets)

        # 2. 添加两列，一列用于一级行业代码，一列用于三级行业代码
        # 初始化这些列的值为'other'
        industry_df['level1'] = 0
        industry_df['level3'] =  0

        # 3. 遍历self.tradeassets，使用字典来填充行业代码
        for stk in self.tradeassets:
            industry_df.at[stk, 'level1'] = swlevel1_dict.get(stk, 0)
            industry_df.at[stk, 'level3'] = swlevel3_dict.get(stk, 0)

        return industry_df
