# Shanfolio


## Modules

- **Datakit** 数据落地和清洗结构
  - DtSql.py 定期调用Sql语句，进行历史数据的下载
  - DtSource.py 数据前处理模块，处理财汇的行情/财务/另类数据，转为矩阵格式，存放历史矩阵数据至数据库。
  - DtLoader.py 数据读取模块，用于读取本地/数据库的矩阵数据
  
- **APT** 因子库，用于存储因子
  - Factor.py 因子的抽象类
  - F_f1.py, F_f2.py, ..

- **FactorMgmt** 因子管理模块
  - FactorParallelCmpt.py 用于并行计算多个因子
  - DtFactorSet.py 因子库管理模块，管理多频率，多类型的因子
  
- **Pred**  基于Qlib的
  - Pred.py 预测
  - Combine.py 合成

- **Backtest** 回测框架
  - Analyzer.py 单因子回测框架，基于Alphalens
  - StgBtVector.py 向量化回测框架
  - StgBt.py 基于Backtrader的回测框架
- **Trade** 交易
- **Monitor** 策略监控

- **Scripts**
  - utils.py 用于统一因子的计算

TODO: 
    Quantlib,
    Learn to Rank algo,
    NLP,
    Genetic Algorithm,
    GAN,
    FinGPT,
    FinRL.
