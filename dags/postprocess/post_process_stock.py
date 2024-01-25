from utils.file_utils import read_pickle_from_sql_tasks


def get_index_info_comp_df_from_local():
    info_index_basic_info_df = read_pickle_from_sql_tasks('info_index_basic_info')
    info_index_basic_info_df.columns = info_index_basic_info_df.columns.str.lower()
    info_index_basic_info_df['symbol'] = info_index_basic_info_df['symbol'].astype(int).astype(str).str.zfill(6)

    info_index_comp_df = read_pickle_from_sql_tasks('info_index_comp')

    info_index_comp_df.columns = info_index_comp_df.columns.str.lower()
    info_index_comp_df['symbol'] = info_index_comp_df['symbol'].astype(int).astype(str).str.zfill(6)
    info_index_comp_df['selectesymbol'] = info_index_comp_df['selectesymbol'].astype(int).astype(str).str.zfill(6)

    return info_index_basic_info_df, info_index_comp_df


# 成分股数据
index_info_df, index_comp_df = get_index_info_comp_df_from_local()
to_chinese_name_dict = dict(zip(index_info_df.symbol, index_info_df.sesname))
to_estclass_dict = dict(zip(index_info_df.symbol, index_info_df.estclass))

# 这是所有的因子 stock factor loader,不同的因子的tradeassets可能不统一
#
# prefix = 'stock'  # TODO enums
#
# basic_info = load_object_from_pickle(f'basic_info_{prefix}')
# tradeassets = load_object_from_pickle(f'tradeassets_{prefix}')
# tradedays = load_object_from_pickle('tradedays')
#
# # 结束#####################################################################################
#
# # 任务1
# size = CNE5.SIZE(prefix).calculate()
# non_linear_size = CNE5.NonLinearSize(prefix).calculate()
# beta = CNE5.BETA(prefix).calculate()
# momentum = CNE5.Momentum(prefix).calculate()
# residual_volatility = CNE5.ResidualVolatility(prefix).calculate()
# liquidity = CNE5.Liquidity(prefix).calculate()
# # 待做
# leverage = CNE5.Leverage(prefix).calculate()
# earnings = CNE5.Earnings(prefix).calculate()
# growth = CNE5.Growth(prefix).calculate()
# dy = CNE5.DY(prefix).calculate()
#
# save_object_to_pickle(liquidity, 'liquidity')
#
# ##
# # 需要为每一个文件定义存放目录，在file_utils中
# factor_list = ['size', 'non_linear_size', 'beta', 'momentum', 'residual_volatility', 'liquidity', 'leverage',
#                'earnings', 'growth', 'dy']
#
# # concat 并获得最近一期数据
# # max_tradedays = max(tradedays)
#
# factor_df_list = []
#
# for i in factor_list:
#     df = load_object_from_pickle(i)
#     # df = df.loc[max_tradedays:, :]
#     df = df.stack()
#     factor_df_list.append(df)
#
# multifactor_df = pd.concat(factor_df_list, axis=1, keys=factor_list)
# multifactor_df = multifactor_df.reset_index()
#
# basic_info_dict = dict(zip(basic_info['symbol'], basic_info['sesname']))
#
# multifactor_df['sesname'] = multifactor_df['symbol'].map(basic_info_dict)
# multifactor_df = multifactor_df[['tradedate', 'symbol', 'sesname'] + factor_list]
#
# visualization_output_path = Path('output') / 'visualization'
# multifactor_df.to_excel(visualization_output_path / 'multifactor.xlsx')
