import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/apple/Documents/Github/SHANFOLIO')
from shanf.apt.Factor import Factor,IndexComp_Factor



class MB(IndexComp_Factor):
    
    def calculate(self,**kwargs):
        
        mb = pd.DataFrame(index =  IndexComp_Factor.fl.__getattr__(kwargs['name']).index, columns = IndexComp_Factor.tradeassets)
        val = IndexComp_Factor.fl.__getattr__(kwargs['name'])
        val = self.prevent_0(val).ffill()

        if kwargs['rank'] == True:
            val = val.rank(axis=1)
        
        for sec,group in IndexComp_Factor.index_comp_dict.items():
            group  = [i for i in group if i in  val.columns.tolist()]
            val_comp = val.loc[:,group]
            
            if kwargs['method'] == 'ma':
                adv_criteria = val_comp - val_comp.rolling(kwargs['period']).mean()
                dcl_criteria = val_comp.rolling(kwargs['period']).mean() - val_comp
                
            if kwargs['method'] == 'net high':
                adv_criteria = val_comp - val_comp.rolling(kwargs['period']).max()
                dcl_criteria = val_comp.rolling(kwargs['period']).min() - val_comp 
                
            if kwargs['method'] == 'adv':
                adv_criteria = val_comp - val_comp.shift(kwargs['period'])
                dcl_criteria = val_comp.shift(kwargs['period']) - val_comp
            
            advancing = np.where(adv_criteria > 0, 1, np.where(adv_criteria<0, 0,np.nan))
            declining = np.where(dcl_criteria > 0, 1, np.where(dcl_criteria>0, 0,np.nan))

            
            # Market Breadth Calculation
            advancing_sum = np.nansum(advancing, axis = 1)
            declining_sum = np.nansum(declining, axis = 1)
            notna_sum = np.count_nonzero(~np.isnan(advancing), axis = 1)
            
            # Market Breadth Ratio
            if kwargs['breadth_method']  == 'total':
                mb.loc[:,sec] = advancing_sum/(notna_sum)
            if kwargs['breadth_method']  == 'net':
                mb.loc[:,sec] = (advancing_sum/declining_sum)
            
        return mb