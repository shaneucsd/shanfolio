import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)

def convert_to_single_quarter(df, numerical_columns):
    # 将累计值转为单季度的值
    df['enddate_quarter'] = df['enddate'].dt.quarter
    enddate_quarter = df.pivot(index='enddate',columns='symbol',values='enddate_quarter')
    
    for col in numerical_columns:
        if col == 'enddate_quarter': continue
        col_df = df.pivot(index='enddate',columns='symbol',values=col)
        if col_df.dtypes.iloc[0]==np.dtype('float64'):
            col_df.loc[:,:] = np.where(col_df==None, np.nan, col_df)
            col_df = pd.DataFrame(np.where(enddate_quarter==1, col_df, (col_df - col_df.shift(1))), index = enddate_quarter.index, columns = enddate_quarter.columns)
            col_df = pd.melt(col_df.reset_index(),id_vars='enddate',value_vars=col_df.columns,value_name=col)
            df = pd.merge(df.drop(col,axis=1), col_df, on=['enddate','symbol'],how='left') 
    return df

def convert_to_ttm(df, numerical_columns):
    df['enddate_quarter'] = df['enddate'].dt.quarter
    enddate_quarter = df.pivot(index='enddate',columns='symbol',values='enddate_quarter')
    
    for col in numerical_columns:
        if col == 'enddate_quarter': continue
        col_df = df.pivot(index='enddate',columns='symbol',values=col).astype(float)

        col_df_last_4q = np.where(
            enddate_quarter == 4, 
            col_df.shift(4),
            np.where(
                enddate_quarter == 3,
                col_df.shift(3),
                np.where(
                    enddate_quarter == 2,
                    col_df.shift(2),
                    col_df.shift(1)
                )
            )
        )

        col_df_yoy = col_df.shift(4)

        my_col_df = np.where(
            ~np.isnan(col_df_yoy)&~np.isnan(col_df_last_4q),
            col_df + col_df_last_4q - col_df_yoy,
            np.where(
                enddate_quarter == 4,
                col_df,
                np.where(enddate_quarter == 3,
                    col_df*(4/3),
                    np.where(
                        enddate_quarter == 2,
                        col_df*(4/2),
                        col_df*4
                    )
                )
            )
        )

        col_df = pd.DataFrame(my_col_df, index = enddate_quarter.index, columns = enddate_quarter.columns)
        col_df = pd.melt(col_df.reset_index(),id_vars='enddate',value_vars=col_df.columns,value_name=col)
        df = pd.merge(df.drop(col,axis=1), col_df, on=['enddate','symbol'],how='left') 
        
    # df = df[df['enddate_quarter']!=4] # remove annual report
    # df.drop('enddate_quarter',axis=1,inplace=True)
    return df

def remove_error_rows(df):
    correct_rows = (
    ((df['enddate'].dt.quarter == 1) & (df['publishdate'].dt.month <= 4) & (df['publishdate'].dt.year == df['enddate'].dt.year)) |
    ((df['enddate'].dt.quarter == 2) & (df['publishdate'].dt.month <= 8) & (df['publishdate'].dt.year == df['enddate'].dt.year)) |
    ((df['enddate'].dt.quarter == 3) & (df['publishdate'].dt.month <= 10) & (df['publishdate'].dt.year == df['enddate'].dt.year)) |
    ((df['enddate'].dt.quarter == 4) & (df['publishdate'].dt.month <= 4) & (df['publishdate'].dt.year == df['enddate'].dt.year + 1))
    )
    error_rows = ~correct_rows


    df.loc[error_rows & (df['enddate'].dt.quarter == 1),'publishdate'] = pd.to_datetime(df.loc[error_rows& (df['enddate'].dt.quarter == 1),'enddate'].dt.year,format="%Y") + pd.DateOffset(months=3, day=30)
    df.loc[error_rows & (df['enddate'].dt.quarter == 2),'publishdate'] = pd.to_datetime(df.loc[error_rows& (df['enddate'].dt.quarter == 2),'enddate'].dt.year,format="%Y") + pd.DateOffset(months=7, day=30)
    df.loc[error_rows & (df['enddate'].dt.quarter == 3),'publishdate'] = pd.to_datetime(df.loc[error_rows& (df['enddate'].dt.quarter == 3),'enddate'].dt.year,format="%Y")+ pd.DateOffset(months=9, day=30)
    df.loc[error_rows & (df['enddate'].dt.quarter == 4),'publishdate'] = pd.to_datetime(df.loc[error_rows& (df['enddate'].dt.quarter == 4),'enddate'].dt.year,format="%Y")+ pd.DateOffset(years=1, months=3, day=30)
    return df

def ensure_continuity(df):
    
    def _ensure_continuity_group(_group):
        # Ensure there is continuous financial report for each symbol
        
        # Determining the start and end dates from the existing data
        start_date = _group['enddate'].min()
        end_date = _group['enddate'].max()

        # Defining the end dates for each quarter using the existing data range
        expected_quarter_end_dates = pd.date_range(start=start_date, end=end_date, freq='Q')

        # Dictionary to map quarter to the corresponding publish date deadline
        publish_date_deadlines = {
            1: pd.DateOffset(months=4, days=-1), # 4.30 for Q1
            2: pd.DateOffset(months=8, days=-1), # 8.30 for Q2
            3: pd.DateOffset(months=10, days=-1), # 10.30 for Q3
            4: pd.DateOffset(months=16, days=-1) # 4.30 next year for Q4
        }
        
        missing_rows = []
        for end_date in expected_quarter_end_dates:
            if end_date not in _group['enddate'].values:
                quarter = (end_date.month - 1) // 3 + 1
                # Creating a row for the missing quarter
                missing_row = pd.DataFrame({
                    'symbol':_group['symbol'].iloc[0],
                    'publishdate': [end_date + publish_date_deadlines[quarter]],
                    'enddate': [end_date],
                    'reportdatetype': [str(quarter)],
                })
                missing_rows.append(missing_row)

        # Concatenating the missing rows with the original dataframe
        if missing_rows:
            _group = pd.concat([_group] + missing_rows, ignore_index=True)

        # Sorting the dataframe by enddate
        _group = _group.sort_values(by='enddate')
        return _group

    df = df.groupby('symbol').apply(lambda group:_ensure_continuity_group(group))
    df = df.droplevel(0)
    df = df.sort_values(['symbol','enddate']).reset_index(drop=True)
    
    return df 