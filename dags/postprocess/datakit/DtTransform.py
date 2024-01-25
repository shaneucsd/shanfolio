from abc import ABC

import numpy as np
import pandas as pd
from postprocess.datakit.DtLoader import DtLoader

from utils.file_utils import load_object_from_pickle


def preprocess(description):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.preprocess_func[func.__name__] = {"description": description}
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class DtTransform(ABC):
    transforms = []

    def __init__(self, prefix="stock", scope="all"):
        self.prefix = prefix
        self.dl = DtLoader(prefix)

        if prefix == "stock":
            self.basic_info = load_object_from_pickle(f"basic_info_{prefix}")
            self.sector_dummy = pd.get_dummies(self.basic_info["swlevel1name"]).astype(
                int
            )
            self.sector_dummy.index = self.basic_info["symbol"].astype(str).values
            self.stock_pool = np.float16(load_object_from_pickle(f"stock_pool_all"))
            self.scope = scope

    def clean(self, factor: pd.DataFrame) -> pd.DataFrame:
        # winsorize and normalize
        if self.prefix == "stock":
            factor = factor * self.stock_pool

        factor.replace(0, np.nan, inplace=True)

        m = np.float64(factor.values)
        m[np.isinf(m)] = np.nan
        iqr = np.percentile(m, 75) - np.percentile(m, 25)
        m = np.clip(
            m, np.percentile(m, 25) - 1.5 * iqr, np.percentile(m, 75) + 1.5 * iqr
        )
        m = (m - np.nanmean(m, axis=1)[:, None]) / (np.nanstd(m, axis=1)[:, None])
        factor = pd.DataFrame(m, index=factor.index, columns=factor.columns)

        return factor

    def winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        def winsorize_row(row):
            median = row.median()
            abs_median = (row - median).abs().median()
            upper_limit = median + 5 * abs_median
            lower_limit = median - 5 * abs_median
            return row.clip(upper=upper_limit, lower=lower_limit)

        return df.apply(winsorize_row, axis=1)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    def replace_zero(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.replace({0: np.nan})

    def fillna_ffill(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ffill()

    def fillna_by_sec_median(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.unstack().reset_index().copy()
        df = df.merge(
            self.basic_info[["symbol", "swlevel1name"]], how="left", on="symbol"
        )
        sec_median = (
            df.groupby(["tradedate", "swlevel1name"])
            .agg({0: lambda x: x.median(skipna=True)})
            .reset_index()
        )
        df = pd.merge(df, sec_median, on=["tradedate", "swlevel1name"], how="left")
        df["0_x"].fillna(df["0_y"], inplace=True)
        df.drop(["0_y"], axis=1, inplace=True)
        df = df.pivot(index="tradedate", columns="symbol", values="0_x")
        return df

    # def neuturalize_sector_and_mktcap(self, df: pd.DataFrame, sector_neu=True, mktcap_neu=True,
    #                                   sm=None) -> pd.DataFrame:
    #     # Actually a Regress of factor on market cap and sector dummy
    #     negotiablemv = self.dl.__getattr__('negotiablemv')  # [tradedate, symbol]

    #     neu_factor = pd.DataFrame(index=df.index, columns=df.columns)

    #     for day in df.index:

    #         factor_data_day = df.loc[day]
    #         market_cap_day = negotiablemv.loc[day]
    #         valid_indices = ~np.isnan(factor_data_day) & ~np.isinf(factor_data_day) & ~np.isnan(
    #             market_cap_day) & ~np.isinf(market_cap_day)

    #         if valid_indices.sum() == 0:
    #             neu_factor.loc[day, :] = np.nan

    #         else:
    #             factor_data_valid = factor_data_day[valid_indices]
    #             market_cap_day_valid = market_cap_day[valid_indices]
    #             stock_sector_valid = self.sector_dummy[valid_indices]

    #             if sector_neu and mktcap_neu:
    #                 neuturalizer = pd.concat([market_cap_day_valid, stock_sector_valid], axis=1)
    #             elif sector_neu and not mktcap_neu:
    #                 neuturalizer = stock_sector_valid
    #             elif not sector_neu and mktcap_neu:
    #                 neuturalizer = market_cap_day_valid
    #             else:
    #                 raise ValueError("At Least one neuturalizer should be applied")

    #             # Regress, get residuals as neutralized factor
    #             neu_factor.loc[day, valid_indices] = sm.OLS(factor_data_valid, neuturalizer).fit().resid

    #     return neu_factor

    def run(self, df, transforms, **kwargs):
        """
        Example:
            trans = DtTransform.DtTransform()
            kwargs = {'neuturalize_sector_and_mktcap':{'sector_neu':True, 'mktcap_neu':True}}
            transforms = [trans.winsorize, trans.normalize,trans.fillna_by_sec_median,trans.neuturalize_sector_and_mktcap]
        """

        for func in transforms:
            df = func(df, **kwargs.get(func.__name__, {}))
        return df
