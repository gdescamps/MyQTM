import codecs
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

import src.config as config


def process_stock(stock, TRADE_END_DATE, TRAIN_START_DATE):
    try:
        stock_news_file = config.DATA_DIR / f"{stock}_{TRADE_END_DATE}_stock_news.json"
        with open(stock_news_file, "r") as f:
            all_news = json.load(f)

        stock_news = {}

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
                if "title" in n:
                    title = n["title"]
                text = None
                if "text" in n:
                    text = n["text"]
                if title is None:
                    pbar.update(1)
                    continue
                if text is None:
                    pbar.update(1)
                    continue

                try:
                    title_decode = codecs.decode(title, "unicode_escape")
                    text_decode = codecs.decode(text, "unicode_escape")
                except Exception as e:
                    # print(f"[{stock}] Exception during decoding: {e}")
                    title_decode = title
                    text_decode = text

                if date not in stock_news:
                    stock_news[date] = []
                stock_news[date].append({"title": title_decode, "text": text_decode})
                pbar.update(1)

        stock_news_file = config.DATA_DIR / f"{stock}_{TRADE_END_DATE}_news.json"
        with open(stock_news_file, "w") as f:
            json.dump(stock_news, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[{stock}] Exception in process_stock: {e}")


def process_all_stocks(config):
    # tqdm.set_lock(threading.RLock())  # Plus nécessaire avec ProcessPoolExecutor
    with tqdm(total=len(config.TRADE_STOCKS), desc="Stocks", position=0) as global_pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    process_stock, stock, config.TRADE_END_DATE, config.TRAIN_START_DATE
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
    if config is None:
        import src.config as config
    process_all_stocks(config)
