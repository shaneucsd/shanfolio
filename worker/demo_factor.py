from db import *

"""
如何读取单个指标
    : dl.get_financial_field_df("salesexpe") 财务
    : dl.close 收盘价
    
"""
close = dl.close
salesexpe = dl.get_financial_field_df("salesexpe")

########################### VALUATION #############################################################
"""

净资产核算的估值
    - PB_LYR
    - 无形资产调整PB
        : 将研发费用、销售费用、管理费用,加到资产负债表中,再计算PB
        - 知识资本(KC)
            - KCi,0 = DEVEEXPE 0 / (g + s)
            - KCi,t = KCi,t-1 * (1 - s) + DEVEEXPE t 
        - 组织资本(OC)
            - OCi,0 = (theta) * (SALEEXP 0 + MANAEXP 0) / (g + s)
            - OCi,t = OCi,t-1 * (1 - s) + (theta) * (SALEEXP t + MANAEXP t)
            - 我们将销售和管理费用的 30%作为组织资本，未资本化的其余 70%的销售和管理费用则视作支持当前而非未来运营的成本
        - 其中,g为股票研发支出的平均增四增长率,s为折旧率,theta为销售管理费用的资本化比例
        
        - 无形资本的计算
            Bi,t INT = B i,t - GDWL i,t + INT i,t
            GDWL为商誉,防止无形资产的重复计算

"""


def get_intangible_asset():
    salesexpe = dl.get_financial_field_df("salesexpe").values
    manaexpe = dl.get_financial_field_df("manaexpe").values
    deveexpe = dl.get_financial_field_df("deveexpe").values
    goodwill = dl.get_financial_field_df("goodwill").values

    salesexpe_1 = dl.get_financial_field_df("salesexpe_1").values
    manaexpe_1 = dl.get_financial_field_df("manaexpe_1").values
    deveexpe_1 = dl.get_financial_field_df("deveexpe_1").values
    goodwill_1 = dl.get_financial_field_df("goodwill_1").values

    totasset = (
        dl.get_financial_field_df("paresharrigh")
        - dl.get_financial_field_df("prest")
        - dl.get_financial_field_df("perbond")
    )
    totasset = totasset.replace(0, np.nan)
    totasset = totasset.values

    theta = 0.3
    sigma_rnd = 0.3
    sigma_sno = 0.2
    g = 0.2

    intasset = pd.DataFrame(index=dl.tradedays, columns=dl.tradeassets)

    first = True
    from tqdm import tqdm

    for symbol in range(len(dl.tradeassets)):
        # for symbol in [76]:
        this_deveexpe = deveexpe[:, symbol]
        this_manaexpe = manaexpe[:, symbol]
        this_salesexpe = salesexpe[:, symbol]
        this_goodwill = goodwill[:, symbol]
        this_deveexpe_1 = deveexpe_1[:, symbol]
        this_manaexpe_1 = manaexpe_1[:, symbol]
        this_salesexpe_1 = salesexpe_1[:, symbol]
        this_goodwill_1 = goodwill_1[:, symbol]

        this_totasset = totasset[:, symbol]

        kc = np.zeros(len(dl.tradedays))
        oc = np.zeros(len(dl.tradedays))
        gw = np.zeros(len(dl.tradedays))
        first = True

        for i, dt in enumerate(dl.tradedays):
            if first and (
                this_deveexpe[i] == 0
                and this_salesexpe[i] == 0
                and this_manaexpe[i] == 0
                and this_goodwill[i] == 0
            ):
                kc[i] = 0
                oc[i] = 0
                gw[i] = 0

            else:
                if first:
                    kc[i] = this_deveexpe[i] / (g + sigma_rnd)
                    oc[i] = (
                        (this_salesexpe[i] + this_manaexpe[i]) * theta / (g + sigma_rnd)
                    )
                    gw[i] = this_goodwill[i]
                    curr_val = (
                        this_deveexpe[i],
                        this_salesexpe[i],
                        this_manaexpe[i],
                        this_goodwill[i],
                    )
                    first = False

                    # print(
                    #     f"dt: {dt}, \n \
                    #             研发费用: {this_deveexpe[i]/1e+8}, SG&A费用 :{(this_salesexpe[i] + this_manaexpe[i])/1e+8}, 商誉: {this_goodwill[i]/1e+8}, \n \
                    #             知识资本: {kc[i]/1e+8}, 经营资本: {oc[i]/1e+8}, 商誉: {gw[i]/1e+8}, 无形资产: {(kc[i]+oc[i]-gw[i])/1e+8}, 总资产: {this_totasset[i]/1e+8} \n \
                    #             无形资产/总资产: {(kc[i]+oc[i]-gw[i])/this_totasset[i]}"
                    # )

                else:
                    if curr_val == (
                        this_deveexpe[i],
                        this_salesexpe[i],
                        this_manaexpe[i],
                        this_goodwill[i],
                    ):
                        kc[i] = kc[i - 1]
                        oc[i] = oc[i - 1]
                        gw[i] = gw[i - 1]
                    else:
                        if dt.month != 4:
                            if this_deveexpe[i] != curr_val[0]:
                                kc[i] = kc[i - 1] * (1 - sigma_rnd) + this_deveexpe[i]
                            if (
                                this_salesexpe[i] != curr_val[1]
                                and this_manaexpe[i] != curr_val[2]
                            ):
                                oc[i] = (
                                    oc[i - 1] * (1 - sigma_rnd)
                                    + (this_salesexpe[i] + this_manaexpe[i]) * theta
                                )
                            if this_goodwill[i] != curr_val[3]:
                                gw[i] = this_goodwill[i]

                            # print(f"dt: {dt},")
                            # print(f"研发费用: {this_deveexpe[i]/1e+8:.2f}, SG&A费用 :{(this_salesexpe[i] + this_manaexpe[i])/1e+8:.2f}, 商誉: {this_goodwill[i]/1e+8:.2f}")

                        elif dt.month == 4:
                            # 将四季报加进来
                            if this_deveexpe[i] != curr_val[0]:
                                kc[i] = (
                                    kc[i - 1] * (1 - sigma_rnd)
                                    + this_deveexpe[i]
                                    + this_deveexpe_1[i]
                                )
                            if (
                                this_salesexpe[i] != curr_val[1]
                                and this_manaexpe[i] != curr_val[2]
                            ):
                                oc[i] = (
                                    oc[i - 1] * (1 - sigma_rnd)
                                    + (this_salesexpe[i] + this_manaexpe[i]) * theta
                                    + (this_salesexpe_1[i] + this_manaexpe_1[i]) * theta
                                )
                            if this_goodwill[i] != curr_val[3]:
                                gw[i] = this_goodwill[i] + this_goodwill_1[i]

                            # print(f"dt: {dt},")
                            # print(f"研发费用: {(this_deveexpe[i]+this_deveexpe_1[i])/1e+8:.2f}, SG&A费用 :{(this_salesexpe[i] + this_manaexpe[i] + this_salesexpe_1[i] + this_manaexpe_1[i])/1e+8:.2f}, 商誉: {(this_goodwill[i]+this_goodwill_1[i])/1e+8:.2f}")
                        else:
                            pass

                        # print(f"知识资本: {kc[i]/1e+8:.2f}, 经营资本: {oc[i]/1e+8:.2f}, 商誉: {gw[i]/1e+8:.2f}, 无形资产: {(kc[i]+oc[i]+gw[i])/1e+8:.2f}, 总资产: {this_totasset[i]/1e+8:.2f}")
                        # print(f"无形资产/总资产: {(kc[i]+oc[i])/this_totasset[i]:.2f}")

                    curr_val = (
                        this_deveexpe[i],
                        this_salesexpe[i],
                        this_manaexpe[i],
                        this_goodwill[i],
                    )

        intasset.iloc[:, symbol] = np.where((kc + oc) > 0, (kc + oc), gw)
    del (
        salesexpe,
        manaexpe,
        deveexpe,
        goodwill,
        salesexpe_1,
        manaexpe_1,
        deveexpe_1,
        goodwill_1,
        totasset,
    )
    return intasset


# 通常需要ffill来填充缺失值
totmktcap = dl.totmktcap.ffill()
# 获得无形资产
netasset = (
    dl.get_financial_field_df("paresharrigh")
    - dl.get_financial_field_df("prest")
    - dl.get_financial_field_df("perbond")
)

# PB
bp = netasset.replace(0, np.nan) / (totmktcap * 10000)
save_object_to_pickle_with_path(bp, "bp", directory + "/valuation")


# 无形资产调整PB
intasset = get_intangible_asset()
bp_int = (netasset + intasset).replace(0, np.nan) / (totmktcap * 10000)
bp_int = bp_int.astype(float)
save_object_to_pickle_with_path(bp_int, "bp_int", directory + "/valuation")

########################### FUND_FLOW #############################################################

"""

挂单资金流向因子：
:大单资金强度(剥离收益率) l_s3_wo_ret
"""

ret20 = dl.close.pct_change(20)

l_s3 = ((dl.linflow - dl.loutflow).rolling(20).sum()) /  ((dl.linflow - dl.loutflow).abs()).rolling(20).sum()
l_s3_wo_ret = panel_reg_residual(l_s3.values, ret20.values)
l_s3_wo_ret = pd.DataFrame(l_s3_wo_ret, index=dl.tradedays, columns=dl.tradeassets)
save_object_to_pickle_with_path(l_s3_wo_ret, "l_s3_wo_ret", directory+ "/fund_flow")



########################### LIQUIDITY #############################################################

"""
市值调整的成交额和换手率
: 成交额 / 市值
: 换手率 / 市值
"""
amt2mv = panel_reg_residual(np.log(dl.amount.values), np.log(dl.negotiablemv.values))
amt2mv = pd.DataFrame(amt2mv, index=dl.tradedays, columns=dl.tradeassets)
save_object_to_pickle_with_path(amt2mv, "amt2mv", directory+ "/liquidity")

tr2mv = panel_reg_residual(np.log(dl.turnrate.values), np.log(dl.negotiablemv.values))
tr2mv = pd.DataFrame(tr2mv, index=dl.tradedays, columns=dl.tradeassets)
save_object_to_pickle_with_path(tr2mv, "tr2mv", directory+ "/liquidity")


