import pandas as pd 
import os


class DtLoader():
    
    """
    DtLoader
    
    
    Read fields    
    
    """
    
    params = {
        'tradedays': None,
        'tradeassets': None,
        'start': '2006-01-01',
        'end': None,
    }
    
    def __init__(self,field_root) -> None:
        """
        
        @field_dict: dict, {field_name: df}, stores all the fields
        @field_root: str, the root path of the fields
        @pathmap: dict, {field_name: (path, file_name)}
        
        """
        
        self.field_dict = {}
        self.field_root = field_root
        self.pathmap = self._map_path(self.field_root) # yield a dict for fields
        self.set_default_days_assets()

    def set_default_days_assets(self):
        self._screen_days = False if self.params['tradedays'] is None else True
        self._screen_assets = False if self.params['tradeassets'] is None else True
        
    def _read_file(self, path, name):        
        df = pd.read_pickle(os.path.join(path, name))
        df.index = pd.to_datetime(df.index)
        return df

    def _map_path(self,path):
        return {f_name: (path, name) for f_name,name in self._yield_path(path)}
    
    def _yield_path(self, path):
        
        for name in os.listdir(path):
            if name.endswith('.pkl'):
                f_name = name.split('.')[0].lower()
                yield f_name, name
            else:
                pass
    
    def get_field_df(self,field):
        def clean_field_df(field_df):
            if self.params['tradeassets'] is not None:
                field_df = field_df.reindex(columns=self.params['tradeassets'])
                
            if self.params['tradedays'] is not None:
                _idx = field_df.index.union(self.params['tradedays'])
                field_df = field_df.reindex(index=_idx).ffill()
                field_df = field_df.loc[self.params['tradedays'], :]
                
            field_df = field_df.loc[self.params['start']:self.params['end'], :]
            return field_df
        
        path, file_name = self.pathmap[field]
        field_df = self._read_file(path, file_name)
        field_df = clean_field_df(field_df)

        return field_df
    
    def validate_field(self,field):
        try:
            name = [ name for name in self.pathmap.keys() if name==field]
            if len(name) == 1:
                return name[0]
        except:
            msg = f'{field} not in the factor list'
            raise ValueError(msg)

    def cached_field(field_func):
        def wrapper(self, field):
            if field not in self.__dict__.keys():
                self.validate_field(field)
                field_df = self.get_field_df(field)
                self.__dict__[field] = field_df
            return field_func(self, field)
        return wrapper
    
    @cached_field
    def __getattr__(self, field):
        return self.get_field_df(field)
    
    def reload(self):
        to_pop = []
        for field in self.__dict__.keys():
            if field in self.pathmap.keys():
                to_pop.append(field)
        for field in to_pop:
            self.__dict__.pop(field)
            