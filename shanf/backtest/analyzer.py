import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np
import stats_utils as su
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy import stats

sns.set_style('darkgrid')


class Analyzer():
    """_summary_

    Data are always store as pd.DataFrame, with tradedate as index, symbol as columns,
    
    Attributes:
        close: pd.DataFrame, close price
        fs : dict, {factor_name: factor_df}
        negotiablemv: pd.DataFrame, negotiablemv
        stock_pool: pd.DataFrame, stock pool

    Returns:
        _type_: _description_
    """
    
    fs = []
    tradeassets = None
    params = {
        'period':'M',
        'period2':'monthly',
        'how':'last',
        'start':None,
        'end':None,
        'figsize':(8,5)    }
    
    
    def __init__(self,close,negotiablemv,fs,stock_pool,**kwargs):
        self.close = close
        self.fs = fs
        self.negotiablemv = negotiablemv
        self.stock_pool = stock_pool
        self.set_params(**kwargs)
        
    def process_info(self):
        # Get Industry Dummy
        basic_info = self.params['basic_info']
        basic_info.dropna(subset=['swlevel1name'],inplace=True)
        
        industry_exposures=pd.DataFrame(pd.get_dummies(basic_info['swlevel1name'])).astype(int)
        industry_exposures.index = basic_info['symbol'].values
        
        self.industry_exposures = industry_exposures
        return industry_exposures
                
    def set_params(self,**kwargs):
        if not kwargs:
            pass
        else:
            self.params.update(kwargs)
    
    def init_bt(self,**kwargs):
        self.set_params(**kwargs)
        
        if self.params['how'] == 'last':
            self.close = self.close.resample(self.params['period']).last()
            
            for key,factor in self.fs.items():
                self.fs[key] = factor.resample(self.params['period']).last()
            
        if self.params['how'] == 'first':
            self.close = self.close.resample(self.params['period']).first()
            for key,factor in self.fs.items():
                self.fs[key] = factor.resample(self.params['period']).first()
                
        self.future_ret = self.close.pct_change().shift(-1)
        self.process_info()
    
    def get_ic(self,name,dir):
        factor = self.fs[name].loc[self.params['start']:self.params['end']] * dir
        future_ret = self.future_ret.loc[self.params['start']:self.params['end']]
        ic = factor.corrwith(future_ret,axis=1,method='spearman')
        return ic 
    
    def eval_ic(self,name,dir=1):
        
        ic = self.get_ic(name,dir)
        
        # IC Mean 
        ic_mean = ic.mean()
        ic_std = ic.std()
        periods = {'M':12,'W':55,'2W':25, 'D':252}
        icir = (ic_mean/ic_std)*np.sqrt(periods[self.params['period']])
        t = ic.mean()*np.sqrt(len(ic.dropna()))/ic.std()
        p = 2*(1-stats.norm.cdf(abs(t)))
        ic_gt_0 = len(ic[ic>0])/len(ic[ic.notna()])
                
        return {
            'ic':ic_mean,
            'ic_std':ic_std,
            'icir':icir,
            't':t,
            'gt_0': ic_gt_0    
        }
        
    def eval_reg(self,name,dir=1,industry_neutral=True, mktcap_neutral=True):
        

        industry_exposures = self.industry_exposures.copy()
        factor = self.fs[name].loc[self.params['start']:self.params['end']] * dir
        future_ret = self.future_ret.loc[self.params['start']:self.params['end']]

        t_values = []
        factor_returns = []

        for date in factor.index[:-1]:
            
            # get next index in factor.index and check if next_date is the same month as date
            next_date = factor.index[factor.index.get_loc(date)+1]
            if next_date.week == date.week or  date not in future_ret.index:
                continue
            
            if date < datetime.datetime.strptime(self.params['start'],'%Y%m%d') or date > datetime.datetime.strptime(self.params['end'],'%Y%m%d'):
                continue
            
            trade_idx = self.stock_pool.index.get_indexer([date],method='ffill')  
            trade_date = self.stock_pool.index[trade_idx]
            available_symbols = self.stock_pool.loc[trade_date].dropna(axis=1).columns.tolist()
            available_symbols = [ i for i in available_symbols if i in industry_exposures.index  ]
            
            next_returns = future_ret.loc[date].loc[available_symbols]
            factor_values = factor.loc[date, available_symbols].to_frame('value')
            
            negotiablemv = self.negotiablemv.loc[trade_date, available_symbols]
            industry_exposures = industry_exposures.loc[available_symbols]
            
            industry_adjusted = industry_exposures.copy()
            weights = negotiablemv.to_numpy().flatten()/negotiablemv.to_numpy().flatten()
            
            for col in industry_adjusted.columns:
                weighted_mean = np.average(industry_adjusted[col], weights=weights)
                industry_adjusted[col] -= weighted_mean

            # Create X matrix including industry exposures and constant
            X = factor_values 
            X = pd.concat([X, industry_adjusted], axis=1)
            X = sm.add_constant(X)
                        
            
            model = sm.WLS(next_returns, X, weights=1.0/np.sqrt(negotiablemv.to_numpy().flatten())).fit()
            
            t_values.append(model.tvalues['value'])
            factor_returns.append(model.params['value'])

        t_abs_mean= np.mean(np.abs(t_values))
        t_gt_2_ratio =  sum(1 for t in np.abs(t_values) if t > 2) / len(np.abs(t_values))
        f_ret_mean = np.mean(factor_returns)
        f_ret_std = np.std(factor_returns)
        f_ret_tv = f_ret_mean / (f_ret_std / np.sqrt(len(factor_returns)))
        t_tv = t_abs_mean / np.std(t_values)

        return {
            't_abs_mean':t_abs_mean,
            't_gt_2_ratio':t_gt_2_ratio,
            'f_ret_mean':f_ret_mean,
            'f_ret_std':f_ret_std,
            'f_ret_tv':f_ret_tv,
            't_tv':t_tv
            
        }        
    def eval_ar(self,name,dir=1):
        
        ic = self.get_ic(name,dir)
        factor = self.fs[name].loc[self.params['start']:self.params['end']] * dir
        future_ret = self.future_ret.loc[self.params['start']:self.params['end']]
        
        ic_ar = ic.corr(ic.shift(1))
        fe_ar = factor.corrwith(factor.shift(1),axis=1).mean()
        
        return {
            'ic_ar':ic_ar,
            'fe_ar':fe_ar
        }
        
        
    def eval_ic_all(self):
        ic_df = pd.DataFrame(columns = ['ic','ic_std','icir','t','gt_0'])
        for name, factor in self.fs.items():
            ic_df.loc[name]=self.eval_ic(name)
        return ic_df
    
    def eval_reg_all(self):
        reg_df = pd.DataFrame(columns = ['t_abs_mean','t_gt_2_ratio','f_ret_mean','f_ret_std','f_ret_tv','t_tv'])
        for name, factor in self.fs.items():
            reg_df.loc[name]=self.eval_reg(name)
        return reg_df
    
    def eval_ar_all(self):
        ar_df = pd.DataFrame(columns = ['ic_ar','fe_ar'])
        for name, factor in self.fs.items():
            ar_df.loc[name]=self.eval_ar(name)
        return ar_df
    
    def eval_perf_all(self):
        perf_df = pd.DataFrame(columns = ['L_ann_pre_ret', 'L_sharpe_ratio', 'LS_ann_pre_ret', 'LS_sharpe_ratio'])
        for name, factor in self.fs.items():
            perf_series = pd.concat([self.eval_perf(name,1).loc['Long',['ann_pre_ret','sharpe_ratio']],
                        self.eval_perf(name,1).loc['Long-Short',['ann_pre_ret','sharpe_ratio']]])
            perf_series.index = [f'L_{name}' if i < 2 else f'LS_{name}' for i, name in enumerate(perf_series.index)]
            perf_df.loc[name]=perf_series
        return perf_df
    
    def get_group_returns(self,name,dir=1,q=5):
        
        factor = self.fs[name].loc[self.params['start']:self.params['end']] * dir
        future_ret = self.future_ret.loc[self.params['start']:self.params['end']]
        
        factor = factor.astype(float)
        future_ret = future_ret.astype(float)
        
        f_quantiles = factor.groupby('tradedate').apply(lambda x: pd.qcut(x.squeeze(), q=q, labels=False, duplicates="drop")+1)
        
        q_ret = pd.DataFrame(columns = [1,2,3,4,5])
        for i in range(1,q+1):
            group = f_quantiles[f_quantiles==i].notna().astype(int)
            q_ret.loc[:,i]=(group*future_ret).sum(axis=1)/group.sum(axis=1)[0]
            
        return q_ret.astype(float)
    
    def get_stgs_returns(self,name,dir=1,q=5):
        factor = self.fs[name].loc[self.params['start']:self.params['end']] * dir
        future_ret = self.future_ret.loc[self.params['start']:self.params['end']]
        q_ret = self.get_group_returns(name,dir,q)
        stg_returns = pd.DataFrame(index=q_ret.index, columns=['Long-Short','Long','Equal Weight'])
        stg_returns['Equal Weight']  = future_ret.mean(axis=1)
        stg_returns['Long-Short'] = q_ret[q] - q_ret[1]
        stg_returns['Long'] = q_ret[q]
        return stg_returns.astype(float)
    
    def eval_perf(self,name,dir,q=5):
        
        q_ret = self.get_group_returns(name,dir,q)
        stg_ret = self.get_stgs_returns(name,dir,q)

        perf = pd.DataFrame() 
        for i in range(1,6):
            perf[i] = su.performance_stats(q_ret[i],stg_ret['Equal Weight'],period=self.params['period2'])
            
        for i in stg_ret.columns:
            perf[i] = su.performance_stats(stg_ret[i],stg_ret['Equal Weight'],period=self.params['period2'])
        return perf.T.round(4)
    
    def plot_q_group(self,name,dir,demeaned=True):
        
        q_ret = self.get_group_returns(name,dir)
        
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        
        if demeaned:
            ax.bar(q_ret.columns,q_ret.mean(axis=0)*np.sqrt(12) - (q_ret.mean(axis=0)*np.sqrt(12)).mean(axis=0))
        else:
            ax.bar(q_ret.columns,q_ret.mean(axis=0)*np.sqrt(12))
            
    def plot_ic_series(self,name,dir,demeaned=True):
        
        ic = self.get_ic(name,dir)
        ic_stats = self.eval_ic(name,dir)

        fig, ax = plt.subplots(figsize=self.params['figsize'])
        
        ax.bar(ic.index,ic.values,width=20,color =[ 'r' if x > 0 else 'g' for x in ic.values],linewidth=0)
        ax.plot(ic.index,ic.rolling(12).mean().values,color='grey')
        # legend with ic, and t value
        ax.set_title(f"{name}, ic = {round(ic_stats['ic'],3)}, t = {round(ic_stats['t'],3)}, ")

        
    def plot_month_ret(self,name,dir,col='Long',q=5):
        
        stg_ret = self.get_stgs_returns(name,dir,q)
        
        plot_df = (stg_ret[col]-stg_ret['Equal Weight']).reset_index()
        plot_df['year'] = plot_df.tradedate.dt.year
        plot_df['month'] = plot_df.tradedate.dt.month
        plot_df = plot_df.pivot(index='year', columns='month', values=0)
        plot_df = plot_df.astype(float)
        
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        sns.heatmap(plot_df, annot=True, fmt=".2%",annot_kws={"size": 8},ax=ax,cmap='RdYlGn_r',center=0)
        plt.yticks(rotation=0)
        
    def plot_q_ret(self,name,dir,demeaned=True):
        
        q_ret = self.get_group_returns(name,dir)
        
        if demeaned:
            q_ret = q_ret.sub(q_ret.mean(axis=1),axis=0)
        else:
            pass
        
        sns.set_palette('tab20')
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        for col in q_ret.columns:        
            sns.lineplot(data=(1+q_ret[col]).cumprod(),ax=ax,label=col)
    
    def plot_stg_ret(self,name,dir,q=5):
        
        stg_ret = self.get_stgs_returns(name,dir,q)
        
        sns.set_palette('tab20')
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        
        for col in ['Long-Short','Long']:
            sns.lineplot(data=(1+stg_ret[col]).cumprod(), ax=ax, label=f"{col}")
        
        for col in ['Long-Short','Long']:
            pre = (1+stg_ret[col]).cumprod()- (1+stg_ret['Equal Weight']).cumprod()
            sns.lineplot(data=pre, ax=ax, label=f"{col} Pre")
            
        sns.lineplot(data=(1+stg_ret['Equal Weight']).cumprod(), ax=ax, label=f"Equal Weight")
    
    
    
    def eval(self):
        ic_df = self.eval_ic_all()
        reg_df = self.eval_reg_all()
        ar_df = self.eval_ar_all()
        perf_df = self.eval_perf_all()
        
        stats_df = pd.concat([perf_df,ic_df,reg_df,ar_df],axis=1)
        # display(stats_df)
        return stats_df

    def plot(self,name,dir=1,demeaned=True):
        perf = self.eval_perf(name,dir)
        display(perf)

        self.plot_q_group(name,dir,demeaned)
        self.plot_q_ret(name,dir,demeaned)
        self.plot_ic_series(name,dir)
        self.plot_stg_ret(name,dir)
        self.plot_month_ret(name,dir)
        
        
        