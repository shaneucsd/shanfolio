# Shanfolio


## Modules

- **dags**

  - **postprocess**
    - **datakit** 数据落地和清洗结构
      - DtSource.py 数据前处理模块，处理财汇的行情/财务/另类数据，转为矩阵格式，存放历史矩阵数据至数据库。
      - DtLoader.py 数据读取模块，用于读取本地/数据库的矩阵数据
      - DtTransform.py 数据清洗模块，行业中性化/ winsorize normalize模块
      
  - **factor_calculation** 因子库
    - Factor.py 因子的抽象类
    - StockFactor.py 股票因子 ， barra因子主要是；
    - 因为是古早的代码 所以比较冗余，计算也慢

  - **utils**  
    - sql、回测设置、numba加速的算子等
    - 在filed_utils里line66改airflow_directory = "./"

  - **enums** 
    - 存储了一部分的enums

  - **backtest** 回测框架
    - analyze_factor.py 单因子回测框架，基于Alphalens

  - **worker** 核心因子生成
    - demo_factor.py 是主要的因子构成代码文件
    - label.py 生成label
    - funcs.py, db.py是用于因子计算函数，数据读取勒的定义
    - bt.ipynb 因子回测、可视化
