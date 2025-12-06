## News 🚀🚀🚀

**2025/12/06: 🔥**

- Stable usage of FMP APIs.
- Corrected data filtering by FMP prioritized for "as reported" data.
- Filtering of news sites prone to revisions.
- Feature ranking by importance using mean / std ^ power, performed in multiple passes.
- Hyperparameter rework to ensure that parameter increases favor safety.
- CMA-ES hyperparameters added to the performance function to prioritize safety.
- Interactive Brokers order execution enabled (paper trading).

## My Quantitative Trading Model (MyQTM)

Welcome to MyQTM, an open-source quantitative trading framework designed to deliver robust, AI-driven investment strategies for high-growth technology stocks. This project demonstrates how to achieve strong annual returns (targeting 50% ROI) by combining structured financial data, unstructured news sentiment, and advanced machine learning techniques.

### Performance Example
![Backtest Performance](backtest.png)

### Key Features

- **Single data provider**: All financial data is sourced from    
    [Financial Modeling Prep](https://site.financialmodelingprep.com/).


![alt text](fmp.png)

- **AI-powered trading decisions**: Integrates time series data, financial indicators, and news sentiment.
- **Robust model evaluation**: Uses interlaced training/testing windows and Monte Carlo backtesting for reliability.
- **Hybrid approach**: Combines classic models (XGBoost) for structured data and LLMs for unstructured news analysis.
- **Stochastic optimization**: Hyperparameters are tuned using Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
- **Transparent methodology**: All code and data processing steps are available for review and validation.

### Watch the Presentation
[![YouTube Video](https://img.youtube.com/vi/c6kNWaCAw6s/0.jpg)](https://youtu.be/c6kNWaCAw6s)

---

> **Disclaimer:** This repository is for educational and research purposes only. It is not financial advice. Use at your own risk and only if you fully understand the algorithms and code.


## Requirements

- **Operating System:** Linux Ubuntu 24.04 or later
- **Python Version:** Python 3.11

Ubuntu 24.04, python 3.11 setup help:
```bash
sudo apt update
sudo apt install -y software-properties-common curl wget build-essential
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

Ubuntu 24.04, git setup help:
```
sudo apt install -y git
git config --global user.name "Your name "
git config --global user.email "your.email@example.com"
```

### Optional Credentials
- **Financial Modeling Prep (FMP):**
	- Register at [https://site.financialmodelingprep.com/](https://site.financialmodelingprep.com/)
	- Obtain your API key for financial data access
    - Add your API key in .env
    ```
    FMP_APIKEY="your API key"
    ```

- **Google Cloud Service Account:**
	- Download a Google Cloud service account JSON file to enable vertexai (Gemini 2.0 Flash) feature from https://cloud.google.com/
    - Add your JSON file path in .env
    ```
    GOOGLE_APPLICATION_CREDENTIALS="./gcp-service-account.json"
    PROJECT_ID="my current project ID"
    ```


### Setup this repo

```
git clone https://github.com/gdescamps/MyQTM.git
cd MyQTM
./1_setup.sh
```

#### What the setup script does

- Creates a dedicated Python 3.11 virtual environment for the project (`venv`)
- Installs all required dependencies
- Automatically downloads financial data, a pretrained model, hyperparameters, and backtest results via DVC (`dvc pull`)

### Run backtest on initial data, pretrained model, and hyperparameters

To run a backtest using the initial financial data, pretrained model, and hyperparameters:

```bash
./5_backtest.sh
```

Results will be saved in the latest folder inside `./outputs/`.

### Prepare a new dataset

To update the dataset to the latest available data:

1. Edit .env add your FMP API key.
2. Edit .env add your GCP service account credential.
3. Edit `src/config.py` and set `TRADE_END_DATE` to today's date.
4. Run the data pipeline script:

```bash
./2_data_pipeline.sh
```

This will fetch and process the latest financial data in `./data/fmp_data`.

### Retrain the model

To retrain the model on the prepared dataset, run:

```bash
./3_train.sh
```

This will update the model and save results in `./outputs/last_train`.

### Search for hyperparameters first PASS

To run hyperparameter optimization (CMA-ES) on the current dataset:

```bash
./4_search_hyperparams.sh
```

This will update the hyperparameters and save results in `./outputs/last_cmaes`.

### Search for hyperparameters second PASS 

To run hyperparameter optimization (CMA-ES) on the current dataset:

1. Edit `src/config.py`
```
CMA_LOOPS = 50  # number of CMA-ES loops
CMA_EARLY_STOP_ROUNDS = 10  # early stopping rounds for CMA-ES
CMA_STOCKS_DROP_OUT_ROUND = 10  # number of dropout rounds for CMA-ES
CMA_STOCKS_DROP_OUT = 5  # number of stocks to drop out during CMA-ES
INIT_X0 = [ # PASTE HYPERPARAMS FROM FIRST PASS    
]  
INIT_CMA_STD = 0.03
```
2. Run the data pipeline script:

```bash
./4_search_hyperparams.sh
```

This will update the hyperparameters and save results in `./outputs/last_cmaes`.


## License

This project is distributed under the **PolyForm Noncommercial License 1.0.0**.

- ✅ Personal, academic, and research use allowed.  
- 🚫 Commercial use (e.g., in proprietary trading systems, financial products, or consulting services) is prohibited without written permission.  
- 💰 For commercial licensing or royalty agreements, contact: [descamps.gregory@gmail.com]

Full license text: [LICENSE](./LICENSE)