class midcap(Factor):
    
    def calculate(self):
        
        midcap = pd.DataFrame(index=tradedays, columns=tradeassets)

        cubic_lncap = f_lncap.factor**3
        for t in f_lncap.factor.index:
            f_lncap_day = f_lncap.factor.loc[t]
            f_cubic_day = cubic_lncap.loc[t]
            totmktcap_day = Factor.fl.totmktcap.loc[t]
            
            # remove nan and inf
            valid_indices= ~np.isnan(f_lncap_day) & ~np.isinf(f_lncap_day) & ~np.isnan(f_cubic_day) & ~np.isinf(f_cubic_day) & ~np.isnan(totmktcap_day) & ~np.isinf(totmktcap_day)
            
            if valid_indices.sum()>0:
                weights = totmktcap_day[valid_indices]
                x = sm.add_constant(f_cubic_day[valid_indices])
                y = f_lncap_day[valid_indices]
                model = sm.WLS(y, x, weights=weights)
                result = model.fit()
                residual = result.resid
                midcap.loc[t, valid_indices] = residual
            else: 
                midcap.loc[t, :] = np.nan
            midcap.index.name= 'tradedate'
            midcap.columns.name='symbol'
        return midcap