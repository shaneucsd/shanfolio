from db import *



"""
Combine features to a df for training.
"""


factors = {
    "valuation":[
        'bp_int'
    ],
    "fund_flow":[
        'l_s3_wo_ret'
    ],
    "liquidity":[
        'amt2mv',
        'tr2mv'
    ],
    "labels":[
        'log_ret_vwap_20d',
    ]
}

# concat to a multi feature dataframe
# df = concat_factor(factors=factors, directory = directory, start="2012-01-01", end='2023-10-01')
# print(df)


"""

Prediction Module for Multi Factor Model

"""

pass






"""
Simply by adding up the factor ranking, we can get a naive multi factor model.

# dt.clean() is a function to winsorize and normalize the factor dataframe.
"""



bp_int = load_object_from_pickle_with_path("bp_int", directory + "/valuation")
l_s3_wo_ret = load_object_from_pickle_with_path("l_s3_wo_ret", directory + "/fund_flow")
tr2mv = load_object_from_pickle_with_path("tr2mv", directory + "/liquidity")

bp_int = dt.clean(bp_int)
l_s3_wo_ret = dt.clean(l_s3_wo_ret)
tr2mv = dt.clean(tr2mv)

bp_int = bp_int.rank(axis=1)
l_s3_wo_ret = l_s3_wo_ret.rank(axis=1)
tr2mv = tr2mv.rank(axis=1)

values = pd.DataFrame(0,index = dl.tradedays,columns = dl.tradeassets)
values += (bp_int + l_s3_wo_ret - tr2mv)
values = values.rank(axis=1)

print(values.tail())

save_object_to_pickle_with_path(values, "naive_multi_factor", directory + "/multi_factor")
