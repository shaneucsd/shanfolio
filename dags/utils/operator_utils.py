import numba as nb
import numpy as np
import pandas as pd
import time


@nb.jit(cache=True)
def ts_mean_1d(a, window=10, minp=None):
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    n = len(a)
    out = np.empty_like(a, dtype=np.float_)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0

    for i in range(n):
        value = a[i]
        if np.isnan(value):
            nancnt += 1
        else:
            cumsum += value

        nancnt_arr[i] = nancnt
        cumsum_arr[i] = cumsum

        window_len = (
            i + 1 - nancnt if i < window else window - (nancnt - nancnt_arr[i - window])
        )
        window_cumsum = cumsum if i < window else cumsum - cumsum_arr[i - window]

        out[i] = window_cumsum / window_len if window_len >= minp else np.nan

    return out


@nb.jit(cache=True, parallel=True)
def ts_mean(matrix, window=10, minp=None):
    n, m = matrix.shape
    result = np.empty_like(matrix, dtype=np.float_)

    for col in nb.prange(m):
        result[:, col] = ts_mean_1d(matrix[:, col], window, minp=minp)

    return result


@nb.jit(cache=True)
def ts_sum_1d(a, window=10, minp=None):
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    n = len(a)
    out = np.empty_like(a, dtype=np.float_)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0

    for i in range(n):
        value = a[i]
        if np.isnan(value):
            nancnt += 1
        else:
            cumsum += value

        nancnt_arr[i] = nancnt
        cumsum_arr[i] = cumsum

        window_len = (
            i + 1 - nancnt if i < window else window - (nancnt - nancnt_arr[i - window])
        )
        window_cumsum = cumsum if i < window else cumsum - cumsum_arr[i - window]

        out[i] = window_cumsum if window_len >= minp else np.nan

    return out


@nb.jit(cache=True, parallel=True)
def ts_sum(matrix, window=10, minp=None):
    n, m = matrix.shape
    result = np.empty_like(matrix, dtype=np.float_)

    for col in nb.prange(m):
        result[:, col] = ts_sum_1d(matrix[:, col], window, minp=minp)

    return result


@nb.jit(cache=True)
def ts_stddev_1d(a, window=10, minp=None, ddof=1):
    """Return rolling standard deviation.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).std(ddof=ddof)`.
    """
    if minp is None:
        minp = window/10
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    cumsum_sq_arr = np.zeros_like(a)
    cumsum_sq = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + a[i]
            cumsum_sq = cumsum_sq + a[i] ** 2
        nancnt_arr[i] = nancnt
        cumsum_arr[i] = cumsum
        cumsum_sq_arr[i] = cumsum_sq
        if i < window:
            window_len = i + 1 - nancnt
            window_cumsum = cumsum
            window_cumsum_sq = cumsum_sq
        else:
            window_len = window - (nancnt - nancnt_arr[i - window])
            window_cumsum = cumsum - cumsum_arr[i - window]
            window_cumsum_sq = cumsum_sq - cumsum_sq_arr[i - window]
        if window_len < minp or window_len == ddof:
            out[i] = np.nan
        else:
            mean = window_cumsum / window_len
            out[i] = np.sqrt(
                np.abs(
                    window_cumsum_sq - 2 * window_cumsum * mean + window_len * mean**2
                )
                / (window_len - ddof)
            )
    return out


@nb.jit(cache=True, parallel=True)
def ts_stddev(matrix, window=10, minp=None, ddof=1):
    n, m = matrix.shape
    result = np.empty_like(matrix, dtype=np.float_)

    for col in nb.prange(m):
        result[:, col] = ts_stddev_1d(matrix[:, col], window, minp, ddof)

    return result


import numba as nb
@nb.jit(nopython = True,cache=True)
def corr_spearman(a,out):
    
	n = len(a)  	#suppose the length of targetValue and out is the same
	aRank = np.argsort(np.argsort(a)) + 1
	bRank = np.argsort(np.argsort(out)) + 1
	d_sum = np.sum((aRank - bRank) ** 2)
	rho_target_out = 1 - (6*d_sum / (n * (n*n -1)))

	return rho_target_out


@nb.jit(cache=True,parallel=True,nopython=True)
def ts_corr(matrix1,matrix2,window=20):
    """
        calculate pairwise correlation, by columns, rolling.
        if matrix2 is a vector, then calculate the correlation between matrix1 and vector
        else if , calculate corr column by column
        
        Example: 
        
        a= np.random.random((100,100))
        b= np.random.random((100,100))
        ts_corr(a,b,window=10)
        
        : Fill nan with 0 by default
    """
    roll_corr = np.zeros_like(matrix1)
    if matrix2.ndim == 1:
        for i in nb.prange(window-1, len(roll_corr)):
            a = matrix1[i-window+1:i+1]
            b = matrix2[i-window+1:i+1]
            for j in nb.prange(a.shape[1]):
                roll_corr[i,j] = corr_spearman(a[:,j],b)
    elif matrix1.shape == matrix2.shape:
        for i in nb.prange(window-1, len(roll_corr)):
            a = matrix1[i-window+1:i+1,:]
            b = matrix2[i-window+1:i+1,:]
            for j in nb.prange(a.shape[1]):
                roll_corr[i,j] = corr_spearman(a[:,j],b[:,j])
    return roll_corr

def ts_cov(matrix, matrix2, window=10):
    # :TODO Update for numba acceleration
    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)
    if isinstance(matrix2, np.ndarray):
        if matrix2.shape[1] == 1:
            matrix2 =  pd.DataFrame(np.repeat(matrix2[:,np.newaxis],matrix.shape[1],axis=1))
        else:
            matrix2 = pd.DataFrame(matrix2)
    
    return matrix.rolling(window,min_periods=window//10).cov(matrix2).values


def ts_rank(matrix, window):
    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)
    return matrix.rolling(window).rank().values

def panel_rank(matrix):
    if isinstance(matrix, np.ndarray):
        is_nan_matrix = np.isnan(matrix)
        matrix = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, matrix)
        matrix = np.where(is_nan_matrix, np.nan, matrix)    
    return matrix


@nb.jit(cache=True)
def ts_prod_1d(a, window=10, minp=None):
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    n = len(a)
    out = np.empty_like(a, dtype=np.float_)
    cumprod_arr = np.ones_like(a, dtype=np.float_)  # 使用累积乘积数组
    nancnt_arr = np.zeros_like(a)
    nancnt = 0

    for i in range(n):
        value = a[i]

        if np.isnan(value):
            nancnt += 1
        else:
            cumprod_arr[i] = cumprod_arr[i - 1] * value if i > 0 else value  # 计算累积乘积

        nancnt_arr[i] = nancnt

        window_len = (
            i + 1 - nancnt if i < window else window - (nancnt - nancnt_arr[i - window])
        )
        window_cumprod = (
            cumprod_arr[i] if i < window else cumprod_arr[i] / cumprod_arr[i - window]
        )

        out[i] = window_cumprod if window_len >= minp else np.nan

    return out


@nb.jit(cache=True, parallel=True)
def ts_prod(matrix, window=10, minp=None):
    # Worked
    n, m = matrix.shape
    result = np.empty_like(matrix, dtype=np.float_)

    for col in nb.prange(m):
        result[:, col] = ts_prod_1d(matrix[:, col], window, minp=minp)

    return result


@nb.jit(cache=True)
def ts_min_1d(a, window, minp=None):
    """Return rolling min.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
    for i in range(a.shape[0]):
        minv = a[i]
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
            if np.isnan(a[j]):
                continue
            if np.isnan(minv) or a[j] < minv:
                minv = a[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@nb.jit(cache=True, parallel=True)
def ts_min(matrix, window, minp=None):
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(matrix, dtype=np.float_)
    for col in nb.prange(matrix.shape[1]):
        out[:, col] = ts_min_1d(matrix[:, col], window, minp=minp)
    return out


@nb.jit(cache=True)
def ts_max_1d(a, window, minp=None):
    """Return rolling max.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    out = np.empty_like(a, dtype=np.float_)
    for i in range(a.shape[0]):
        max_value = np.nan
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
            if not np.isnan(a[j]) and (np.isnan(max_value) or a[j] > max_value):
                max_value = a[j]
            cnt += 1
        if cnt < minp or max_value == np.nan:
            out[i] = np.nan
        else:
            out[i] = max_value
    return out


@nb.jit(cache=True, parallel=True)
def ts_max(a, window, minp=None):
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in nb.prange(a.shape[1]):
        out[:, col] = ts_max_1d(a[:, col], window, minp=minp)
    return out


@nb.jit(cache=True)
def diff_1d(a, window):
    """Return the 1-th discrete difference.

    Numba equivalent to `pd.Series(a).diff()`."""
    out = np.empty_like(a, dtype=np.float_)
    out[:n] = np.nan
    out[n:] = a[n:] - a[:-n]
    return out


@nb.jit(cache=True)
def delta(matrix, window):
    """2-dim version of `diff_1d_nb`."""
    out = np.empty_like(matrix, dtype=np.float_)
    for col in nb.prange(matrix.shape[1]):
        out[:, col] = diff_1d(matrix[:, col], window=window)
    return out


# Define a Numba-compiled function to compute argmax
@nb.jit(cache=True)
def ts_argmax_1d(a, window, minp=None):
    """Return rolling max.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).apply(np.argmax)`.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    out = np.empty_like(a, dtype=np.float_)
    for i in range(a.shape[0]):
        max_idx = -1
        max_value = np.nan
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
            if not np.isnan(a[j]) and (np.isnan(max_value) or a[j] > max_value):
                max_idx = j + 1
                max_value = a[j]
            cnt += 1
        if cnt < minp or max_idx == -1:
            out[i] = np.nan
        else:
            out[i] = max_idx
    return out


# Define the Numba-accelerated ts_argmax function
@nb.jit(cache=True, parallel=True)
def ts_argmax(matrix, window, minp=None):
    out = np.empty_like(matrix, dtype=np.float_)
    for col in nb.prange(matrix.shape[1]):
        out[:, col] = ts_argmax_1d(matrix[:, col], window=window, minp=minp)
    return out


# Define a Numba-compiled function to compute argmin
@nb.jit(cache=True)
def ts_argmin_1d(a, window, minp=None):
    """Return rolling min.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).apply(np.argmin)`.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")

    out = np.empty_like(a, dtype=np.float_)
    for i in range(a.shape[0]):
        min_idx = -1
        min_value = np.nan
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
            if not np.isnan(a[j]) and (np.isnan(min_value) or a[j] < min_value):
                min_idx = j + 1
                min_value = a[j]
            cnt += 1
        if cnt < minp or min_idx == -1:
            out[i] = np.nan
        else:
            out[i] = min_idx
    return out


# Define the Numba-accelerated ts_argmin function
@nb.jit(cache=True, parallel=True)
def ts_argmin(matrix, window, minp=None):
    out = np.empty_like(matrix, dtype=np.float_)
    for col in nb.prange(matrix.shape[1]):
        out[:, col] = ts_argmin_1d(matrix[:, col], window=window, minp=minp)
    return out


@nb.jit(cache=True)
def ewm_mean_1d(a, span, minp, adjust):
    """Return exponential weighted average.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp, adjust=adjust).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments.
    """
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(a)
    out = np.empty(N, dtype=np.float_)
    if N == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha
    weighted_avg = a[0]
    is_observation = weighted_avg == weighted_avg
    nobs = int(is_observation)
    out[0] = weighted_avg if (nobs >= minp) else np.nan
    old_wt = 1.0

    for i in range(1, N):
        cur = a[i]
        is_observation = cur == cur
        nobs += is_observation
        if weighted_avg == weighted_avg:
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (
                        old_wt + new_wt
                    )
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.0
        elif is_observation:
            weighted_avg = cur
        out[i] = weighted_avg if (nobs >= minp) else np.nan
    return out


@nb.jit(cache=True)
def ewm_mean(matrix, span, minp=None, adjust=False):
    """2-dim version of `ewm_mean_1d_nb`."""
    out = np.empty_like(matrix, dtype=np.float_)
    for col in range(matrix.shape[1]):
        out[:, col] = ewm_mean_1d(matrix[:, col], span, minp=minp, adjust=adjust)
    return out

@nb.jit(cache=True,forceobj=True)
def window_sum(x, w):
    c = np.cumsum(x)
    s = c[w - 1 :]
    s[1:] -= c[:-w]
    return s


@nb.jit(cache=True,forceobj=True)
def window_lin_reg(y, x, w):
    # 计算各种滚动窗口和
    sx = window_sum(x, w)
    sy = window_sum(y, w)
    sx2 = window_sum(x**2, w)
    sxy = window_sum(x * y, w)

    n = w  # 滚动窗口的大小

    slope = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    intercept = (sy - slope * sx) / n
    return slope, intercept

@nb.jit(cache=True)
def ts_reg_1d(a, b, window=10):
    # 暂时仅支持带截距的回归
    slope, intercept = window_lin_reg(a, b, window)
    slope = np.concatenate([np.array([np.nan] * (window - 1)), slope])
    intercept = np.concatenate([np.array([np.nan] * (window - 1)), intercept])
    return slope, intercept

@nb.jit(cache=True)
def ts_reg_resi_1d(a, b, window=10):
    # 暂时仅支持带截距的回归
    slope, intercept = window_lin_reg(a, b, window)
    slope = np.concatenate([np.array([np.nan] * (window - 1)), slope])
    intercept = np.concatenate([np.array([np.nan] * (window - 1)), intercept])
    residual = a - slope * b - intercept
    return residual

@nb.jit(cache=True, parallel=True)
def ts_reg_slope(matrix, bm, window=10):
    out = np.empty_like(matrix, dtype=np.float_)
    matrix = np.nan_to_num(matrix)
    bm = np.nan_to_num(bm)
    if bm.ndim == 1:
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_1d(matrix[:, col], bm, window)[0]
    elif bm.shape == matrix.shape: 
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_1d(matrix[:, col], bm[:, col], window)[0]
    else:
        raise ValueError("bm must be a 1D array or a 2D array with the same shape as matrix")
    return out

@nb.jit(cache=True, parallel=True,forceobj=True)
def ts_reg_intercept(matrix, bm, window=10):
    out = np.empty_like(matrix, dtype=np.float_)
    matrix = np.nan_to_num(matrix)
    bm = np.nan_to_num(bm)
    if bm.ndim == 1:
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_1d(matrix[:, col], bm, window)[1]
    elif bm.shape == matrix.shape: 
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_1d(matrix[:, col], bm[:, col], window)[1]
    else:
        raise ValueError("bm must be a 1D array or a 2D array with the same shape as matrix")
    return out

@nb.jit(cache=True, parallel=True,forceobj=True)

def ts_reg_residual(matrix, bm, window=10):
    # bm is a [n,1] array or a [n,m] matrix
    # matrix is a [n,m] matrix
    out = np.empty_like(matrix, dtype=np.float_)
    matrix = np.nan_to_num(matrix)
    bm = np.nan_to_num(bm)
    if bm.ndim == 1:
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_resi_1d(matrix[:, col], bm, window)
    elif bm.shape == matrix.shape: 
        for col in nb.prange(matrix.shape[1]):
            out[:, col] = ts_reg_resi_1d(matrix[:, col], bm[:, col], window)
    else:
        raise ValueError("bm must be a 1D array or a 2D array with the same shape as matrix")
    return out

@nb.jit(cache=True)
def panel_reg_resi_1d(y, x):
    vld = ~np.isnan(x) & ~np.isnan(y) 
    valid_percentage = np.sum(vld) / len(x)
    if valid_percentage < 0.05:
        return np.full_like(y, np.nan) 
    x_v = x[vld]
    y_v = y[vld]
    x_wc = np.column_stack((np.ones(len(x_v)), x_v))
    XTX = np.linalg.inv(np.dot(x_wc.T, x_wc))
    XTy = np.dot(x_wc.T, y_v)
    beta = np.dot(XTX, XTy)
    r = np.empty_like(y)
    r[vld] = y_v - np.dot(x_wc, beta)
    r[~vld] = np.nan
    return r

@nb.jit(cache=True, parallel=True,forceobj=True)
def panel_reg_residual(matrix, matrix2):
    # matrix is a [n,m] matrix
    # matrix2 is another [n,m] matrix
    # regress each row of matrix on the corresponding row of matrix2
    # Here I would not use nan to num as it would change the result 
    out = np.empty_like(matrix, dtype=np.float_)
    assert matrix.shape == matrix2.shape
    for row in nb.prange(matrix.shape[0]):
        out[row,:] = panel_reg_resi_1d(matrix[row,:], matrix2[row, :])
    return out

@nb.jit(cache=True, parallel=True)
def ts_coef_var(matrix, window):
    out = np.empty_like(matrix, dtype=np.float_)
    for col in nb.prange(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            start_idx = max(i - window + 1, 0)
            end_idx = i + 1
            values = matrix[start_idx:end_idx, col]
            valid_values = values[~np.isnan(values)]

            if len(valid_values) < window:
                out[i, col] = np.nan
            else:
                mean_val = np.nanmean(valid_values)
                std_val = np.nanstd(valid_values)
                out[i, col] = std_val / mean_val

    return out

def panel_binning(matrix, q=5):
    out = np.empty_like(matrix, dtype=np.float_)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        valid_values = row[~np.isnan(row)]
        quantiles = np.nanpercentile(valid_values, np.linspace(0, 100, q + 1))
        try:
            labels = np.digitize(valid_values, quantiles, right=False)
            labels[labels == q+1] = q
            out[i, ~np.isnan(row)] = labels
            out[i, np.isnan(row)] = np.nan
        except:
            out[i, :] = np.nan    
    return out
