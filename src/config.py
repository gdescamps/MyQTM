"""
config.py

This module contains configuration variables and constants used throughout the project. These include dates, directories, stock lists, optimization parameters, and other settings.

Variables:
- BASE_END_DATE: The base end date for data processing.
- TRADE_END_DATE: The end date for trading operations.
- FINETUNE_END_DATE: The end date for fine-tuning operations.
- TRADE_START_DATE: The start date for trading operations.
- TRAIN_START_DATE: The start date for training operations.
- TRAIN_END_DATE: The end date for training operations.
- TEST_START_DATE: The start date for testing operations.
- TEST_END_DATE: The end date for testing operations.
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

from skopt.space import Real

BASE_END_DATE_FILE = "2025-09-05"
BASE_END_DATE = "2025-09-05"

# TRADE_END_DATE = "2025-09-05"
TRADE_END_DATE = "2025-11-28"  # base on 10-22
FINETUNE_END_DATE = "2025-04-05"

if TRADE_END_DATE == BASE_END_DATE_FILE:
    BASE_END_DATE_FILE = None
    BASE_END_DATE = None

TRADE_START_DATE = "2017-01-01"
TRAIN_START_DATE = "2019-08-01"
TRAIN_END_DATE = "2024-10-05"

TEST_START_DATE = "2020-01-03"
INITIAL_CAPITAL = 8800

TEST_END_DATE = TRADE_END_DATE

DATA_DIR = "./data/"
TRAIN_DIR = "./outputs/last_train"
CMA_DIR = "./outputs/last_cma"

INDICES = ["^IXIC", "^VIX"]

RELIABLE_NEWS_SITES = [
    "247wallst.com",  # 156/7766 (2.01% erreurs, 1.7% des news)
    "barrons.com",  # 88/7910 (1.11% erreurs, 1.7% des news)
    "benzinga.com",  # 30/15881 (0.19% erreurs, 3.4% des news)
    "businessinsider.com",  # 22/4625 (0.48% erreurs, 1.0% des news)
    "cnbc.com",  # 130/17219 (0.75% erreurs, 3.7% des news)
    "cnet.com",  # 4/3042 (0.13% erreurs, 0.7% des news)
    "cnn.com",  # 0/2405 (0.00% erreurs, 0.5% des news)
    "deadline.com",  # 0/1047 (0.00% erreurs, 0.2% des news)
    "etftrends.com",  # 0/571 (0.00% erreurs, 0.1% des news)
    "fastcompany.com",  # 0/1021 (0.00% erreurs, 0.2% des news)
    "feedproxy.google.com",  # 0/841 (0.00% erreurs, 0.2% des news)
    "finbold.com",  # 64/3581 (1.79% erreurs, 0.8% des news)
    "fool.com",  # 888/72676 (1.22% erreurs, 15.6% des news)
    "forbes.com",  # 98/10540 (0.93% erreurs, 2.3% des news)
    "foxbusiness.com",  # 1/1890 (0.05% erreurs, 0.4% des news)
    "fxempire.com",  # 0/888 (0.00% erreurs, 0.2% des news)
    "geekwire.com",  # 20/1707 (1.17% erreurs, 0.4% des news)
    "gurufocus.com",  # 0/4171 (0.00% erreurs, 0.9% des news)
    "investopedia.com",  # 0/5166 (0.00% erreurs, 1.1% des news)
    "investorplace.com",  # 335/34688 (0.97% erreurs, 7.5% des news)
    "investors.com",  # 74/6436 (1.15% erreurs, 1.4% des news)
    "invezz.com",  # 80/4314 (1.85% erreurs, 0.9% des news)
    "marketbeat.com",  # 0/4718 (0.00% erreurs, 1.0% des news)
    "markets.businessinsider.com",  # 0/1011 (0.00% erreurs, 0.2% des news)
    "marketwatch.com",  # 136/16753 (0.81% erreurs, 3.6% des news)
    "news.sky.com",  # 3/692 (0.43% erreurs, 0.1% des news)
    "newsfilecorp.com",  # 20/1832 (1.09% erreurs, 0.4% des news)
    "nypost.com",  # 10/3546 (0.28% erreurs, 0.8% des news)
    "nytimes.com",  # 8/1731 (0.46% erreurs, 0.4% des news)
    "proactiveinvestors.co.uk",  # 0/3180 (0.00% erreurs, 0.7% des news)
    "proactiveinvestors.com",  # 8/3978 (0.20% erreurs, 0.9% des news)
    "pulse2.com",  # 0/928 (0.00% erreurs, 0.2% des news)
    "pymnts.com",  # 0/4534 (0.00% erreurs, 1.0% des news)
    "reuters.com",  # 31/17771 (0.17% erreurs, 3.8% des news)
    "schaeffersresearch.com",  # 6/2277 (0.26% erreurs, 0.5% des news)
    "seekingalpha.com",  # 360/34919 (1.03% erreurs, 7.5% des news)
    "stockmarket.com",  # 0/1329 (0.00% erreurs, 0.3% des news)
    "techcrunch.com",  # 74/5984 (1.24% erreurs, 1.3% des news)
    "techxplore.com",  # 0/2579 (0.00% erreurs, 0.6% des news)
    "thedogofwallstreet.com",  # 16/535 (2.99% erreurs, 0.1% des news)
    "theguardian.com",  # 0/1682 (0.00% erreurs, 0.4% des news)
    "thelincolnianonline.com",  # 0/1118 (0.00% erreurs, 0.2% des news)
    "venturebeat.com",  # 4/468 (0.85% erreurs, 0.1% des news)
    "youtube.com",  # 12/10323 (0.12% erreurs, 2.2% des news)
    "zacks.com",  # 12/51714 (0.02% erreurs, 11.1% des news)
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

MAX_POSITIONS = 12
CMA_RECURSIVE = 1
CMA_LOOPS = 150
CMA_EARLY_STOP_ROUNDS = 30
CMA_STOCKS_DROP_OUT_ROUND = 20
CMA_STOCKS_DROP_OUT = 10
CMA_PROCESSES = 128
CMA_PARALLEL_PROCESSES = 32
INIT_X0 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
INIT_CMA_STD = 0.2

INIT_SPACE = [
    Real(0.01, 0.999, name="long_open_prob_thres_A"),
    Real(0.01, 0.999, name="long_close_prob_thres_A"),
    Real(0.01, 0.999, name="short_open_prob_thres_A"),
    Real(0.01, 0.999, name="short_close_prob_thres_A"),
    Real(0.01, 0.999, name="long_open_prob_thres_B"),
    Real(0.01, 0.999, name="long_close_prob_thres_B"),
    Real(0.01, 0.999, name="short_open_prob_thres_B"),
    Real(0.01, 0.999, name="short_close_prob_thres_B"),
    Real(0.01, 0.999, name="increase_positions_count"),
]
PARAM_GRID = {
    "patience": [100],
    "max_depth": [8],
    "learning_rate": [0.01],
    "subsample": [0.6],
    "colsample_bytree": [0.7],
    "gamma": [4],
    "min_child_weight": [5],
    "reg_alpha": [0.4],
    "reg_lambda": [4],
    "mean_std_power": [1.71],
    "top_features": list(range(55, 85, 1)),
}

TRADE_DATA_LOAD = None
DATES_PORTFOLIO = []

ENABLE_PROFILER = False
