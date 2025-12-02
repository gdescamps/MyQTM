import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

from src.config import TRADE_STOCKS
from src.utils.llm import save_llm_cache, txt2txt_llm
from src.utils.path import get_project_root


def parse_llm_sentiment_response(response):
    """
    Parse LLM response and extract sentiment scores for company, competitors, and global.
    Returns a dict with default value 5 if not found.
    """
    sentiment = {"company": 5, "competitors": 5, "global": 5}
    if not response:
        return sentiment

    # Regex to find lines like 'company: 7'
    pattern = re.compile(r"(company|competitors|global)\s*:\s*(\d+)", re.IGNORECASE)
    for match in pattern.finditer(response):
        key = match.group(1).lower()
        value = int(match.group(2))
        if key in sentiment and 0 <= value <= 10:
            sentiment[key] = value
    return sentiment


def process_stock(stock, config):

    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Load company profile data
    profile_file = data_path / f"{stock}_{config.TRADE_END_DATE}_profile.json"
    profile_file = str(profile_file)

    try:
        with open(profile_file, "r") as f:
            profile = json.load(f)
    except Exception as e:
        return

    sentiments = {}
    # Load news data for the stock
    news_file = data_path / f"{stock}_{config.TRADE_END_DATE}_news.json"
    news_file = str(news_file)

    try:
        with open(news_file, "r") as f:
            news = json.load(f)
    except Exception as e:
        return

    keys = list(news.keys())
    # Add progress bar for processing news items
    for idx, key in enumerate(tqdm(keys, desc=f"{stock} news", leave=False), start=1):
        news_item = news[key]
        news_text = ""
        for n in news_item:
            if "title" not in n:
                continue
            if "text" not in n:
                continue
            if n["title"] is None:
                continue
            if n["text"] is None:
                continue

            news_text += n["title"] + ": " + n["text"] + "\n"
        # Build prompt for LLM to extract sentiment scores
        news_text += f"From this news related to the company {profile['companyName']} provide 3 sentiments score from 0 to 10, 0 is extremly negative, 5 neutral and 10 extremy posisitive, the sentiment is related to the value of the compagny, the  \n"
        news_text += f"Based on this news about {profile['companyName']}, provide 3 sentiment scores from 0 to 10 (0 = extremely negative, 5 = neutral, 10 = extremely positive). The scores should reflect the potential impact on the company's value. Please rate:\n"
        news_text += "company: The company's products and revenues\n"
        news_text += "competitors: The company's competitors\n"
        news_text += "global: Global factors (politics, taxes, regulations, wars, diseases, elections)\n"
        news_text += "provide only output like :\n"
        news_text += "company: 7\n"
        news_text += "competitors: 4\n"
        news_text += "global: 5\n"
        news_text += "if there's no news about the category return 5\n"
        sentiment = {"company": 5, "competitors": 5, "global": 5}
        if len(news_text) > 0:
            try:
                # Call LLM to analyze sentiment from news text
                prompt, _ = txt2txt_llm(
                    news_text, model_name="gemini-2.0-flash", cache=True
                )
                sentiment = parse_llm_sentiment_response(prompt)
                sentiments[key] = sentiment
            except Exception as e:
                print(f"Error processing sentiment for {stock} news item {key}: {e}")
                sentiments[key] = {"company": 5, "competitors": 5, "global": 5}

    # Save sentiment scores to file
    sentiments_file = news_file.replace(".json", "_sentiments.json")
    # Save any remaining cache at the end
    save_llm_cache()
    try:
        with open(sentiments_file, "w") as f:
            json.dump(sentiments, f)
    except Exception as e:
        print(f"Error saving sentiments for {stock}: {e}")
        return


def process_all_stocks(config):
    # Process all stocks in parallel using thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            tqdm(
                executor.map(partial(process_stock, config=config), TRADE_STOCKS),
                total=len(TRADE_STOCKS),
            )
        )


def main(config=None):
    # Load default config if not provided
    if config is None:
        import src.config as config
    process_all_stocks(config)
