
import pandas as pd
import numpy as np
from tqdm import tqdm


"""
存储了一些单个因子会临时使用的函数
"""

def align_index(field_x, field_y):

    return field_x, field_y

def adjust_to_quarter(df):
    df.loc[ ~(((df.index.month==4)) | \
                ((df.index.month==8)) | \
                ((df.index.month==10)) )] = np.nan
    return df

def align_publish(x,y):
    x = x.ffill()
    x[np.isnan(y)] = np.nan
    return x,y


def calc_linear_residual(field_y, field_x, dl, method = 'reg', window = 8):
    
    common_index = field_y.index.union(field_x.index)
    field_y = field_y.reindex(common_index)
    field_x = field_x.reindex(common_index)
    field = pd.Series(index = common_index)


    # Preallocate residuals array
    res = np.full(shape=(len(common_index),), fill_value=np.nan)  

    symbols = np.array([idx[1] for idx in common_index])

    # Iterate over unique symbols only once
    for symbol in dl.tradeassets:
        symbol_mask = symbols == symbol

        # Extract the data for the current symbol
        y_values = field_y[symbol_mask].ffill().values
        x_values = field_x[symbol_mask].ffill().values
        residuals = res[symbol_mask]

        # Calculate residuals where enough data is available
        for i in range(len(y_values) - window + 1):
            y_window = y_values[i:i + window]
            x_window = x_values[i:i + window]

            if method == 'z_reg' or method == 'reg' or method == 'reg_resi':            
            
                if method == 'z_reg':
                    y_window = (y_window - np.mean(y_window)) / np.std(y_window)
                    x_window = (x_window - np.mean(x_window)) / np.std(x_window)
                
                if not np.isnan(x_window).any() and not np.isnan(y_window).any():
                    p = np.polyfit(x_window, y_window, 1)
                    predicted = np.polyval(p, x_window)
                    
                    if method == 'reg_resi':
                        residuals[i + window - 1] = np.nanstd(y_window - predicted) 
                    else:
                        residuals[i + window - 1] = y_window[-1] - predicted[-1] 

            if method == 'slope':
                if not np.isnan(x_window).any() and not np.isnan(y_window).any():
                    p = np.polyfit(x_window, y_window, 1)
                    residuals[i + window - 1] = p[0]

            if method == 'qoq':
                y_qoq = (y_values[i] - y_values[i-1])/y_values[i-1]
                x_qoq = (x_values[i] - x_values[i-1])/x_values[i-1] 
                
                if not np.isnan(x_qoq) and not np.isnan(y_qoq):
                    residuals[i] = y_qoq - x_qoq

        res[symbol_mask] = residuals

    field = pd.Series(data=res, index=common_index).unstack()    
    field = field.reindex(index=dl.tradedays, columns=dl.tradeassets).ffill()

    return field

def calc_back_forth_reg_residual(field_y, field_x, dl, method = 'reg', window = 8):
    common_index = field_y.index.union(field_x.index)
    field_y = field_y.reindex(common_index)
    field_x = field_x.reindex(common_index)
    field = pd.Series(index = common_index)
    res = np.full(shape=(len(common_index),), fill_value=np.nan)  

    symbols = np.array([idx[1] for idx in common_index])


    for symbol in dl.tradeassets:
        symbol_mask = symbols == symbol

        # Extract the data for the current symbol
        y_values = field_y[symbol_mask].ffill().values
        x_values = field_x[symbol_mask].ffill().values
        std_residuals = res[symbol_mask]

        # Calculate residuals where enough data is available
        for i in range(1,len(y_values) - window-1):
            y_window = y_values[i:i + window]
            x_b1 = x_values[i-1:i + window-1]
            x    = x_values[i:i+window]
            x_f1 = x_values[i+1:i + window+1]

            # finish the code here.
            if not np.isnan(y_window).any() and not np.isnan(x).any() and not  \
                np.isnan(x_b1).any() and not np.isnan(x_f1).any()  :
                # Perform linear regression
                X_window = np.column_stack((np.ones_like(x), x, x_b1, x_f1))
                coefficients = np.linalg.lstsq(X_window, y_window, rcond=None)[0]

                # Calculate predicted values
                y_pred = np.dot(X_window, coefficients)

                # Calculate residuals
                residuals = y_window - y_pred

                # Calculate the standard deviation of residuals
                std_residuals[i] = np.nanstd(residuals)

        res[symbol_mask] = residuals

    field = pd.Series(data=res, index=common_index).unstack()    
    field = field.reindex(index=dl.tradedays, columns=dl.tradeassets).ffill()


def calc_normalize( field_x, dl, window = 6):
    common_index = field_x.index
    res = np.full(shape=(len(common_index),), fill_value=np.nan)  

    symbols = np.array([idx[1] for idx in common_index])

    # Iterate over unique symbols only once
    for symbol in dl.tradeassets:
        symbol_mask = symbols == symbol

        # Extract the data for the current symbol
        x = field_x[symbol_mask].ffill()
        res[symbol_mask] =  (x - x.rolling(window).mean()) / x.rolling(window).std()

    field = pd.Series(data=res, index=common_index).unstack()    
    field = field.reindex(index=dl.tradedays, columns=dl.tradeassets).ffill()

    return field


