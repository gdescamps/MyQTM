from pathlib import Path

from skopt.space import Real

from src.utils.path import get_project_root

# Path setup
DATA_DIR = Path(get_project_root()) / "data" / "fmp_data"
TRAIN_DIR = "./outputs/last_train"
HYPERPARAMS_DIR = "./outputs/last_cmaes"

# Dates setup
TEST_START_DATE = "2020-01-03"  # start of backtesting period
INITIAL_CAPITAL = 8800  # initial capital for backtesting same as nasdaq index value at TEST_START_DATE to compare both
TRADE_END_DATE = "2025-10-22"  # end of backtesting period
TEST_END_DATE = TRADE_END_DATE
TRADE_START_DATE = "2017-01-01"
TRAIN_START_DATE = "2019-08-01"  # start of training period
TRAIN_END_DATE = "2024-10-05"  # end of training period

# Symbols list
INDICES = ["^IXIC", "^VIX"]
COMMODITIES = ["GCUSD", "CLUSD"]
TRADE_STOCKS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "NFLX",
    "AZN",
    "NVDA",
    "TSLA",
    "EA",
    "META",
    "AMZN",
    "BA",
    "AMD",
    "INTC",
    "IBM",
    "PEP",
    "PYPL",
    "AVGO",
    "QCOM",
    "CSCO",
    "DIS",
    "NKE",
    "PFE",
    "BABA",
    "VRTX",
    "REGN",
    "ILMN",
    "BIIB",
    "SCHW",
    "INTU",
    "FITB",
    "TDG",
    "CTAS",
    "ODFL",
    "CPRT",
    "DLTR",
    "ROST",
    "MNST",
    "KDP",
    "ENPH",
    "SEDG",
    "PLUG",
    "FSLR",
    "ADSK",
    "CDNS",
    "WDAY",
    "ALB",
    "TECH",
    "CRWD",
    "ZS",
    "MDB",
    "ESTC",
    "MRNA",
    "NBIX",
    "EXAS",
    "INCY",
    "FATE",
    "ALNY",
    "NVAX",
    "ASML",
    "ON",
    "LRCX",
    "MPWR",
    "MKSI",
    "FORM",
    "AMBA",
    "SLAB",
    "AEHR",
    "RUN",
    "BLDP",
    "BE",
    "NOVT",
    "ROKU",
    "TTD",
    "LYFT",
    "BKNG",
    "ETSY",
    "CHWY",
    "MTCH",
    "MELI",
    "PINS",
    "CAT",
    "LMT",
    "JPM",
    "MA",
    "XOM",
    "RIO",
    "PG",
    "KO",
    "VZ",
    "NVO",
    "TSM",
    "SAP",
    "RGLD",
    "CRM",
    "SHOP",
    "DQ",
    "CWEN",
    "AES",
    "SPOT",
    "HEI",
    "GE",
    "AGX",
    "UBER",
    "BSX",
    "MDT",
    "EFX",
]
TRADE_STOCKS = list(set(TRADE_STOCKS))
XGBOOST_HYPERPARAM_GRID_SEARCH = {
    "patience": [100],
    "max_depth": [7],
    "learning_rate": [0.01],
    "subsample": [0.6],
    "colsample_bytree": [0.7],
    "gamma": [4],
    "min_child_weight": [5],
    "reg_alpha": [0.4],
    "reg_lambda": [4],
    "top_features": [200],
}  # hyperparameter grid for XgBoost tuning
TS_SIZE = 6  # common time series size
OPEN_DELAY = 1  # day delay to keep signal alive
MAX_POSITIONS = 12  # maximum open positions at a time
CMA_LOOPS = 200  # number of CMA-ES loops
CMA_EARLY_STOP_ROUNDS = 30  # early stopping rounds for CMA-ES
CMA_STOCKS_DROP_OUT_ROUND = 1  # number of dropout rounds for CMA-ES
CMA_STOCKS_DROP_OUT = 0  # number of stocks to drop out during CMA-ES
INIT_X0 = [
    0.6,
    0.3,
    0.6,
    0.3,
    0.6,
    0.3,
    0.6,
    0.3,
    0.5,
    0.5,
    0.5,
    0.5,
]  # initial solution for CMA-ES
INIT_CMA_STD = 0.2  # initial standard deviation for CMA-ES
INIT_SPACE = [
    Real(0.2, 0.99, name="long_open_prob_thresa"),
    Real(0.01, 0.99, name="long_close_prob_thresa"),
    Real(0.2, 0.99, name="short_open_prob_thresa"),
    Real(0.01, 0.99, name="short_close_prob_thresa"),
    Real(0.2, 0.99, name="long_open_prob_thresb"),
    Real(0.01, 0.99, name="long_close_prob_thresb"),
    Real(0.2, 0.99, name="short_open_prob_thresb"),
    Real(0.01, 0.99, name="short_close_prob_thresb"),
    Real(0.05, 1.0, name="long_prob_powera"),
    Real(0.05, 1.0, name="short_prob_powera"),
    Real(0.05, 1.0, name="long_prob_powerb"),
    Real(0.05, 1.0, name="short_prob_powerb"),
]  # search space for CMA-ES
TRADE_DATA_LOAD = None  # global variable to cache loaded trade data
