from backtest.analyze_factor import Analyzer
from postprocess.factor_calculation import StockFactor, IndexFactor


params = {
    "prefix": "stock",
    "freq": "D",
    "n_quantiles": 10,
    "start": "2022-01-01",
    "end": "2023-01-01",
    "direction": 1,
    "benchmark_symbol": "000300",
    "scope": "000852",
    "commission_rate": 0.000,
    "slippage_rate": 0.000,
}


analyzer = Analyzer(**params)
analyzer.set_target_factor('gene1')

analyzer.get_indicators()