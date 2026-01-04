"""
config.py

This module contains configuration variables and constants used throughout the project. These include dates, directories, stock lists, optimization parameters, and other settings.

Variables:
- BASE_END_DATE: The base end date for data processing.
- TRAIN_START_DATE: The start date for training operations.
- TRAIN_END_DATE: The end date for training operations.
- TEST_START_DATE: The start date for testing operations.
- INITIAL_CAPITAL: The initial capital for trading simulations.
- DATA_DIR: Directory for storing data files.
- TRAIN_DIR: Directory for storing training outputs.
- CMA_DIR: Directory for storing CMA-ES outputs.
- INDICES: List of indices used in the project.
- RELIABLE_NEWS_SITES: List of reliable news sources for data collection.
- TRADE_MAIN_STOCKS: List of main stocks for trading.
- TRADE_GROWTH_STOCKS: List of growth stocks for trading.
- TRADE_VALUE_STOCKS: List of value stocks for trading.
- NEW_CANDIDATE_STOCKS: List of new candidate stocks for trading.
- TRADE_STOCKS: Combined list of all stocks for trading.
- TS_SIZE: Time series size for data processing.
- MAX_POSITIONS: Maximum positions allowed overall.
- CMA_RECURSIVE: Number of recursive iterations for CMA-ES.
- CMA_LOOPS: Number of loops for CMA-ES optimization.
- CMA_EARLY_STOP_ROUNDS: Early stopping rounds for CMA-ES.
- CMA_STOCKS_DROP_OUT_ROUND: Number of stock dropout rounds for CMA-ES.
- CMA_STOCKS_DROP_OUT: Number of stocks to drop out per round.
- CMA_PROCESSES: Number of processes for CMA-ES.
- CMA_PARALLEL_PROCESSES: Number of parallel processes for CMA-ES.
- INIT_X0: Initial parameter values for CMA-ES.
- INIT_CMA_STD: Initial standard deviation for CMA-ES.
- INIT_SPACE: Parameter space for CMA-ES optimization.
- PARAM_GRID: Grid of parameters for model training.
- TRADE_DATA_LOAD: Placeholder for trade data loading configuration.
- DATES_PORTFOLIO: Placeholder for portfolio dates.
- ENABLE_PROFILER: Flag to enable or disable profiling.
"""

from datetime import datetime

from skopt.space import Real

BASE_END_DATE_FILE = "2025-09-05"
BASE_END_DATE = "2025-09-05"

BENCHMARK_START_DATE = "2020-01-03"
# BENCHMARK_END_DATE = "2025-09-05"
BENCHMARK_END_DATE = "2025-12-22"

if BENCHMARK_END_DATE == BASE_END_DATE_FILE:
    BASE_END_DATE_FILE = None
    BASE_END_DATE = None

DOWNLOAD_START_DATE = "2017-01-01"
TRAIN_START_DATE = "2019-08-01"
TRAIN_END_DATE = "2024-10-05"

# CMAES_END_DATE = TRAIN_END_DATE
# CMAES_END_DATE = BENCHMARK_END_DATE
CMAES_END_DATE = "2025-03-05"  # CMA-ES finetune end date
CMAES_MONTHS_HISTORY = 3 * 12
# CMAES_START_DATE = None
CMAES_START_DATE = BENCHMARK_START_DATE

if CMAES_END_DATE is not None and CMAES_START_DATE is None:
    cmaes_end_dt = datetime.strptime(CMAES_END_DATE, "%Y-%m-%d")
    total_months = cmaes_end_dt.year * 12 + (cmaes_end_dt.month - 1)
    target_months = total_months - CMAES_MONTHS_HISTORY
    target_year = target_months // 12
    target_month = (target_months % 12) + 1
    try:
        cmaes_start_dt = cmaes_end_dt.replace(
            year=target_year,
            month=target_month,
        )
    except ValueError:
        # Handle end-of-month truncation (e.g., 31 -> 30/28).
        cmaes_start_dt = cmaes_end_dt.replace(
            year=target_year,
            month=target_month,
            day=28,
        )
    CMAES_START_DATE = cmaes_start_dt.strftime("%Y-%m-%d")

INITIAL_CAPITAL = (
    8800  # Manual capital init to sync portforlio and nasdaq at BENCHMARK_START_DATE
)

DATA_DIR = "./data/"
TRAIN_DIR = "./outputs/last_train"
CMA_DIR = "./outputs/last_cma"

INDICES = ["^IXIC", "^VIX"]

# Important this is website selected for quality of news
# We keep only news sources that are not fixed after publication
# So we could consider these news as reported
RELIABLE_NEWS_SITES = [
    "247wallst.com",  # 156/7766 (2.01% errors, 1.7% of news)
    "barrons.com",  # 88/7910 (1.11% errors, 1.7% of news)
    "benzinga.com",  # 30/15881 (0.19% errors, 3.4% of news)
    "businessinsider.com",  # 22/4625 (0.48% errors, 1.0% of news)
    "cnbc.com",  # 130/17219 (0.75% errors, 3.7% of news)
    "cnet.com",  # 4/3042 (0.13% errors, 0.7% of news)
    "cnn.com",  # 0/2405 (0.00% errors, 0.5% of news)
    "deadline.com",  # 0/1047 (0.00% errors, 0.2% of news)
    "etftrends.com",  # 0/571 (0.00% errors, 0.1% of news)
    "fastcompany.com",  # 0/1021 (0.00% errors, 0.2% of news)
    "feedproxy.google.com",  # 0/841 (0.00% errors, 0.2% of news)
    "finbold.com",  # 64/3581 (1.79% errors, 0.8% of news)
    "fool.com",  # 888/72676 (1.22% errors, 15.6% of news)
    "forbes.com",  # 98/10540 (0.93% errors, 2.3% of news)
    "foxbusiness.com",  # 1/1890 (0.05% errors, 0.4% of news)
    "fxempire.com",  # 0/888 (0.00% errors, 0.2% of news)
    "geekwire.com",  # 20/1707 (1.17% errors, 0.4% of news)
    "gurufocus.com",  # 0/4171 (0.00% errors, 0.9% of news)
    "investopedia.com",  # 0/5166 (0.00% errors, 1.1% of news)
    "investorplace.com",  # 335/34688 (0.97% errors, 7.5% of news)
    "investors.com",  # 74/6436 (1.15% errors, 1.4% of news)
    "invezz.com",  # 80/4314 (1.85% errors, 0.9% of news)
    "marketbeat.com",  # 0/4718 (0.00% errors, 1.0% of news)
    "markets.businessinsider.com",  # 0/1011 (0.00% errors, 0.2% of news)
    "marketwatch.com",  # 136/16753 (0.81% errors, 3.6% of news)
    "news.sky.com",  # 3/692 (0.43% errors, 0.1% of news)
    "newsfilecorp.com",  # 20/1832 (1.09% errors, 0.4% of news)
    "nypost.com",  # 10/3546 (0.28% errors, 0.8% of news)
    "nytimes.com",  # 8/1731 (0.46% errors, 0.4% of news)
    "proactiveinvestors.co.uk",  # 0/3180 (0.00% errors, 0.7% of news)
    "proactiveinvestors.com",  # 8/3978 (0.20% errors, 0.9% of news)
    "pulse2.com",  # 0/928 (0.00% errors, 0.2% of news)
    "pymnts.com",  # 0/4534 (0.00% errors, 1.0% of news)
    "reuters.com",  # 31/17771 (0.17% errors, 3.8% of news)
    "schaeffersresearch.com",  # 6/2277 (0.26% errors, 0.5% of news)
    "seekingalpha.com",  # 360/34919 (1.03% errors, 7.5% of news)
    "stockmarket.com",  # 0/1329 (0.00% errors, 0.3% of news)
    "techcrunch.com",  # 74/5984 (1.24% errors, 1.3% of news)
    "techxplore.com",  # 0/2579 (0.00% errors, 0.6% of news)
    "thedogofwallstreet.com",  # 16/535 (2.99% errors, 0.1% of news)
    "theguardian.com",  # 0/1682 (0.00% errors, 0.4% of news)
    "thelincolnianonline.com",  # 0/1118 (0.00% errors, 0.2% of news)
    "venturebeat.com",  # 4/468 (0.85% errors, 0.1% of news)
    "youtube.com",  # 12/10323 (0.12% errors, 2.2% of news)
    "zacks.com",  # 12/51714 (0.02% errors, 11.1% of news)
]

TRADE_MAIN_STOCKS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "NFLX",
    "AZN",
    "NVDA",
    # "ABNB",
    "TSLA",
    "EA",
    "META",
    "AMZN",
    "BA",
    "AMD",
    "INTC",
    "IBM",
    "PEP",
]
TRADE_GROWTH_STOCKS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "NFLX",
    # "AZN",
    "NVDA",
    # "ABNB",
    "PYPL",
    "TSLA",
    "EA",
    "META",
    "AMZN",
    "BA",
    "AMD",
    "INTC",
    "IBM",
    "PEP",
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
    # "LCID",
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
    # "ANSS",
    "WDAY",
    "ALB",
    "TECH",
    # # 🖥️ Technologie / Logiciels / Cloud
    "CRWD",  # CrowdStrike - cybersécurité
    # "DDOG",  # Datadog - monitoring cloud
    # "SNOW",  # Snowflake - entrepôt de données
    "ZS",  # Zscaler - sécurité cloud
    "MDB",  # MongoDB - base de données
    # "PLTR",  # Palantir - big data analytique
    # "U",  # Unity - moteur 3D / jeux
    "ESTC",  # Elastic - search, observabilité
    # "APP",  # AppLovin - publicité mobile
    # "AI",  # C3.ai - intelligence artificielle
    # 💉 Santé / Biotech / Pharma
    "MRNA",  # Moderna - vaccins
    "NBIX",  # Neurocrine Biosciences
    "EXAS",  # Exact Sciences - diagnostic
    "INCY",  # Incyte - oncologie
    "FATE",  # Fate Therapeutics - cellules souches
    "ALNY",  # Alnylam - thérapies ARN
    "NVAX",  # Novavax - vaccins
    # # 🧬 Semi-conducteurs / Hardware
    "ASML",  # ASML - lithographie EUV
    "ON",  # Onsemi - semi pour véhicules
    "LRCX",  # Lam Research
    "MPWR",  # Monolithic Power Systems
    "MKSI",  # MKS Instruments
    "FORM",  # FormFactor
    "AMBA",  # Ambarella - AI vidéo
    "SLAB",  # Silicon Labs - IoT
    # "AEHR",  # Aehr Test Systems
    # "ACLX",  # AcelRx - tech médicale
    # # 🌱 Énergie / Environnement / Green Tech
    "RUN",  # Sunrun - solaire
    "BLDP",  # Ballard Power
    "BE",  # Bloom Energy
    # "ARRY",  # Array Technologies
    # "SHLS",  # Shoals Tech
    "NOVT",  # Novanta
    # # 🛍️ Consommation / Services / Divertissement
    "ROKU",  # Roku - streaming
    "TTD",  # The Trade Desk./
    "LYFT",  # Lyft - mobilité
    "BKNG",  # Booking Holdings
    "ETSY",  # Etsy - e-commerce
    "CHWY",  # Chewy - animaux
    "MTCH",  # Match Group
    "MELI",  # MercadoLibre
    "PINS",  # Pinterest
    # "PARA",  # Paramount
    # "AGX",  # infrastructures énergétiques pour data centers, notamment pour l’IA.
    # "UBER",  # Uber - mobilité
    # "BSX",
    # "MDT",
    # "EFX",
]

TRADE_VALUE_STOCKS = [
    "CAT",  # – Caterpillar (infrastructures, construction, machines lourdes)
    "LMT",  # – Lockheed Martin (défense, aérospatial → secteur peu corrélé à la tech)
    "JPM",  # – JPMorgan Chase (banque universelle, solide bilan)
    "MA",  # – Mastercard (paiements mondiaux, complément à la tech e-commerce que tu as déjà)
    "XOM",  # – ExxonMobil (énergie traditionnelle, dividende défensif)
    "RIO",  # – Rio Tinto (mines, métaux – utile comme hedge contre inflation et croissance émergente)
    "PG",  # – Procter & Gamble (produits de base, stabilité)
    "KO",  # – Coca-Cola (marque forte, consommation récurrente)
    "VZ",  # – Verizon (télécom US, rendement défensif)
    "NVO",  # – Novo Nordisk (Danemark, leader mondial diabète/obésité – complément santé biotech US)
    # "TSM",  # – TSMC (Taiwan, semi mais avec poids Asie, différent de NVDA/AMD car plus manufacturier)
    "SAP",  # – SAP (logiciels, Allemagne → diversification hors US)
    # "TFPM", # métaux précieux, or/argent.
    # "FNV",
    # "RGLD",
]

NEW_CANDIDATE_STOCKS = [
    # "FNV",  # Franco-Nevada
    "RGLD",  # Royal Gold
    # "WPM",  # Wheaton Precious Metals
    # "TFPM",  # Triple Flag Precious Metals
    # "NEM",  # Newmont Corporation
    # "GOLD",  # Barrick Gold
    # "AEM",  # Agnico Eagle Mines
    # "KGC",  # Kinross Gold
    # 🖥️ Technologie / Cloud / IA
    "CRM",  # Salesforce - SaaS / cloud
    # "NET",  # Cloudflare - cybersécurité & edge computing
    "SHOP",  # Shopify - e-commerce SaaS
    # "OKTA",  # Okta - gestion des identités
    # "ZSAN",  # Zscaler (déjà mais je laisse NET et OKTA pour cloud security)
    # 💉 Santé / Biotech
    # "DXCM",  # DexCom - dispositifs médicaux (diabète)
    # "VEEV",  # Veeva Systems - logiciels cloud pour biotech/pharma
    # "IDXX",  # Idexx Labs - diagnostic vétérinaire
    # "GH",  # Guardant Health - diagnostic oncologique
    # "CRSP",  # CRISPR Therapeutics - thérapies géniques
    # "EDIT",  # Editas Medicine - thérapie génétique
    # 🧬 Semi / Hardware complémentaires
    # "NVMI",  # Nova Ltd. - métrologie semi
    # 🌱 Green / Énergie
    "DQ",  # Daqo New Energy - solaire (polysilicium)
    # "NOVA",  # Sunnova - solaire résidentiel
    "CWEN",  # Clearway Energy - renouvelable
    "AES",  # AES Corporation - renouvelable & batteries
    # 🛍️ Services & Conso
    # "SE",  # Sea Limited - e-commerce & gaming Asie
    # "DASH",  # DoorDash - livraison
    # "COIN",  # Coinbase - crypto / fintech
    # "SQ",  # Block (ex-Square) - fintech
    # "AFRM",  # Affirm - paiement fractionné
    "SPOT",  # Spotify - streaming
    # ✈️ Industriels / Défense / Infra
    "HEI",  # Heico - composants aéronautiques
    "GE",  # GE Aerospace - moteurs & défense
    # "NOC",  # Northrop Grumman - défense
    "AGX",  # infrastructures énergétiques pour data centers, notamment pour l’IA.
    "UBER",  # Uber - mobilité
    "BSX",
    "MDT",
    "EFX",
]

TRADE_STOCKS = list(
    set(TRADE_GROWTH_STOCKS + TRADE_VALUE_STOCKS + NEW_CANDIDATE_STOCKS)
)

TS_SIZE = 6


# CMA-ES optimization parameters
CMA_RECURSIVE = 1
CMA_LOOPS = 150
CMA_EARLY_STOP_ROUNDS = 30
CMA_STOCKS_DROP_OUT_ROUND = 20
CMA_STOCKS_DROP_OUT = 10
CMA_PROCESSES = 32
CMA_PARALLEL_PROCESSES = 16
TREND_SCORE_RATE = 0.6
INIT_X0 = [
    0.8,
    0.33,
    0.8,
    0.33,
    0.8,
    0.33,
    0.8,
    0.33,
    0.85,
    0.85,
    0.1,
    0.1,
    0.1,
    TREND_SCORE_RATE,
]
INIT_CMA_STD = 0.2
# CMA-ES optimization parameter space
INIT_SPACE = [  # Important increase params favor better safety
    Real(0.33, 0.95, name="long_open_prob_thres_A"),
    Real(0.1, 0.95, name="long_close_prob_thres_A"),
    Real(0.33, 0.95, name="short_open_prob_thres_A"),
    Real(0.1, 0.95, name="short_close_prob_thres_A"),
    Real(0.33, 0.95, name="long_open_prob_thres_B"),
    Real(0.1, 0.95, name="long_close_prob_thres_B"),
    Real(0.33, 0.95, name="short_open_prob_thres_B"),
    Real(0.1, 0.95, name="short_close_prob_thres_B"),
    Real(0.6, 0.95, name="long_pos_count"),
    Real(0.6, 0.95, name="short_pos_count"),
    Real(0.03, 0.2, name="long_pos_pow"),
    Real(0.03, 0.2, name="short_pos_pow"),
    Real(0.05, 0.2, name="pos_gain_close_thres"),
    Real(0.0, 0.99, name="trend_score_rate"),
]
CMA_PARAM_NAMES = [dim.name for dim in INIT_SPACE]

# XGBoost model training parameter grid
PARAM_GRID = {
    "patience": [100],
    "max_depth": [7],
    "learning_rate": [0.01],
    "subsample": [0.6],
    "colsample_bytree": [0.7],
    "gamma": [4],
    "min_child_weight": [5],
    "reg_alpha": [0.4],
    "reg_lambda": [4],
    # Important parameter for mean/(std^power)
    # search power value that provides
    # best feature importance rank to maximize F1
    "mean_std_power": [1.71],
    # Top features search range
    "top_features": list(range(55, 65, 1)),
}

F1_THRESHOLD_STEP = 0.0001
OPEN_INDEX_DELAY = 1
NEW_OPEN = True
TRADE_DATA_LOAD = None
DATES_PORTFOLIO = []

ENABLE_PROFILER = False
