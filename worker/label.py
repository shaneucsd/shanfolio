from db import *


########################### Labels #############################################################

dl.vwap = dl.vwap.ffill()

log_ret_vwap = np.log(dl.vwap).diff(1).shift(-1)
log_ret_vwap_5d = np.log(dl.vwap).diff(5).shift(-5)
log_ret_vwap_20d = np.log(dl.vwap).diff(20).shift(-20)

dl.close = dl.close.ffill()

log_ret_close = np.log(dl.close).diff(1).shift(-1)
log_ret_close_5d = np.log(dl.close).diff(5).shift(-5)
log_ret_close_20d = np.log(dl.close).diff(20).shift(-20)

# normalize the return by panel mean

log_ret_vwap = log_ret_vwap - np.nanmean(log_ret_vwap.values)
log_ret_vwap_5d = log_ret_vwap_5d - np.nanmean(log_ret_vwap_5d.values)
log_ret_vwap_20d = log_ret_vwap_20d - np.nanmean(log_ret_vwap_20d.values)

save_object_to_pickle_with_path(log_ret_vwap, "log_ret_vwap", directory+ "/labels")
save_object_to_pickle_with_path(log_ret_vwap_5d, "log_ret_vwap_5d", directory+ "/labels")
save_object_to_pickle_with_path(log_ret_vwap_20d, "log_ret_vwap_20d", directory+ "/labels")

log_ret_close = log_ret_close - np.nanmean(log_ret_close.values)
log_ret_close_5d = log_ret_close_5d - np.nanmean(log_ret_close_5d.values)
log_ret_close_20d = log_ret_close_20d - np.nanmean(log_ret_close_20d.values)

save_object_to_pickle_with_path(log_ret_close, "log_ret_close", directory+ "/labels")
save_object_to_pickle_with_path(log_ret_close_5d, "log_ret_close_5d", directory+ "/labels")
save_object_to_pickle_with_path(log_ret_close_20d, "log_ret_close_20d", directory+ "/labels")


print(log_ret_vwap.tail())