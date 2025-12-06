import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import src.config as config
from src.path import get_project_root


def process_stock(stock, TRADE_END_DATE, TRAIN_START_DATE):

    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    try:
        # Load stock news data
        stock_news_file = data_path / f"{stock}_{TRADE_END_DATE}_stock_news.json"
        with open(stock_news_file, "r") as f:
            all_news = json.load(f)

        # Merge with base historical data if available
        if config.BASE_END_DATE_FILE is not None:

            base_stock_news_file = (
                data_path / f"{stock}_{config.BASE_END_DATE_FILE}_stock_news.json"
            )

            with open(base_stock_news_file, "r") as f:
                base_stock_news = json.load(f)

            base_end_date = pd.to_datetime(config.BASE_END_DATE)

            # Filter news after base end date
            after_base_data = []
            for item in all_news:
                item_date = pd.to_datetime(item["publishedDate"])
                if item_date > base_end_date:
                    after_base_data.append(item)

            all_news = after_base_data + base_stock_news

        stock_news = {}

        # Filter news after training start date
        all_news_ = []
        for n in all_news:
            date = pd.to_datetime(n["publishedDate"])
            if date > pd.to_datetime(TRAIN_START_DATE):
                all_news_.append(n)

        all_news = all_news_

        # Progress bar for news items of this stock
        with tqdm(
            total=len(all_news),
            desc=f"{stock}",
            leave=False,
        ) as pbar:
            for n in all_news:
                date = pd.to_datetime(n["publishedDate"])
                date = date.strftime("%Y-%m-%d")
                title = None
                text = None
                site = None
                if "title" in n:
                    title = n["title"]
                text = None
                if "text" in n:
                    text = n["text"]
                if "site" in n:
                    site = n["site"]
                    # Skip news from unreliable sources
                    if site not in config.RELIABLE_NEWS_SITES:
                        pbar.update(1)
                        continue
                if title is None:
                    pbar.update(1)
                    continue
                if text is None:
                    pbar.update(1)
                    continue
                if site is None:
                    pbar.update(1)
                    continue

                def has_only_ascii(value):
                    # Check if string contains only ASCII characters
                    try:
                        value.encode("ascii")
                        return True  # Only ASCII
                    except UnicodeEncodeError:
                        return False  # Contains non-ASCII characters

                def keep_only_ascii(value):
                    # Remove non-ASCII characters from string
                    return value.encode("ascii", errors="ignore").decode("ascii")

                # Clean and validate title
                try:
                    assert isinstance(title, str)
                    title_checked = keep_only_ascii(title)
                    assert has_only_ascii(title_checked)
                except Exception as e:
                    print(
                        f"[{stock}, {n['publishedDate']}] Exception during cleaning: {e}"
                    )
                    title_checked = ""

                # Clean and validate text
                try:
                    assert isinstance(text, str)
                    text_checked = keep_only_ascii(text)
                    assert has_only_ascii(text_checked)
                except Exception as e:
                    print(
                        f"[{stock}, {n['publishedDate']}] Exception during cleaning: {e}"
                    )
                    print(f"  Original text: {repr(text[:100])}")
                    text_checked = ""

                if date not in stock_news:
                    stock_news[date] = []

                stock_news[date].append({"title": title_checked, "text": text_checked})
                pbar.update(1)

        # Save processed news to file
        stock_news_file = data_path / f"{stock}_{TRADE_END_DATE}_news.json"
        with open(stock_news_file, "w") as f:
            json.dump(stock_news, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[{stock}] Exception in process_stock: {e}")


def process_all_stocks(config):
    # Process all stocks in parallel using ProcessPoolExecutor
    with tqdm(total=len(config.TRADE_STOCKS), desc="Stocks", position=0) as global_pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    process_stock,
                    stock,
                    config.BENCHMARK_END_DATE,
                    config.TRAIN_START_DATE,
                ): stock
                for stock in config.TRADE_STOCKS
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception in process: {e}")
                global_pbar.update(1)


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
