import itertools

import numpy as np
import pandas as pd


def safe_div(a, b):
    return np.where((b != 0) & (~np.isnan(b)), a / b, np.nan)


def generate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df_ratios = df.copy()

    def add_ratio(name, num, denom):
        if {num, denom}.issubset(df.columns):
            df_ratios[name] = safe_div(df[num], df[denom])

    # 1. Multi-horizon momentum and volatility
    add_ratio("momentum_1d", "ts_d_close_0", "ts_d_close_1")
    add_ratio("momentum_3d", "ts_d_close_0", "ts_d_close_3")
    add_ratio("momentum_5d", "ts_d_close_0", "ts_d_close_5")
    add_ratio("momentum_1w", "ts_w_close_0", "ts_w_close_1")
    add_ratio("momentum_3w", "ts_w_close_0", "ts_w_close_3")
    add_ratio("momentum_5w", "ts_w_close_0", "ts_w_close_5")

    add_ratio("vol_momentum_3d", "ts_d_volume_0", "ts_d_volume_3")
    add_ratio("vol_momentum_5d", "ts_d_volume_0", "ts_d_volume_5")
    add_ratio("vol_momentum_5w", "ts_w_volume_0", "ts_w_volume_5")

    # price/volume ratio = market conviction intensity
    add_ratio("price_vs_volume", "ts_d_close_0", "ts_d_volume_0")

    # 2. Technical indicators
    ema_pairs = [
        ("ema_4", "ema_8"),
        ("ema_8", "ema_30"),
        ("ema_30", "ema_50"),
        ("ema_50", "ema_200"),
        ("ema_8", "ema_200"),
    ]
    for short, long in ema_pairs:
        add_ratio(f"{short}_over_{long}", short, long)

    add_ratio("ema4_vs_rsi", "ema_4", "rsi_14")
    add_ratio("rsi_over_price", "rsi_14", "ts_d_close_0")
    add_ratio("rsi_over_ema200", "rsi_14", "ema_200")
    add_ratio("rsi_over_vix", "rsi_14", "vix_ema_50")
    add_ratio("ema_slope_ratio", "ema_8", "ema_8")  # placeholder for future derivatives

    # 3. Macro and rates
    add_ratio("inflation_vs_rate", "ts_ei_inflationRate_0", "ts_ei_federalFunds_0")
    add_ratio("CPI_vs_rate", "ts_ei_CPI_0", "ts_ei_federalFunds_0")
    add_ratio("retail_vs_inflation", "ts_ei_retailSales_0", "ts_ei_inflationRate_0")
    add_ratio(
        "industrial_vs_inflation",
        "ts_ei_industrialProductionTotalIndex_0",
        "ts_ei_inflationRate_0",
    )
    add_ratio(
        "consumerSentiment_vs_inflation",
        "ts_ei_consumerSentiment_0",
        "ts_ei_inflationRate_0",
    )
    add_ratio(
        "recessionProb_vs_rate",
        "ts_ei_smoothedUSRecessionProbabilities_0",
        "ts_ei_federalFunds_0",
    )
    add_ratio(
        "unemployment_vs_inflation", "ts_ei_unemploymentRate_0", "ts_ei_inflationRate_0"
    )
    add_ratio(
        "housing_vs_rate",
        "ts_ei_newPrivatelyOwnedHousingUnitsStartedTotalUnits_0",
        "ts_ei_federalFunds_0",
    )
    add_ratio("durableGoods_vs_rate", "ts_ei_durableGoods_0", "ts_ei_federalFunds_0")

    # inflation/rate spread = macro spread
    df_ratios["inflation_minus_rate"] = (
        df["ts_ei_inflationRate_0"] - df["ts_ei_federalFunds_0"]
    )

    # 4. Inter-market ratios
    add_ratio("gold_vs_nasdaq", "gcusd_ema_50", "ixic_ema_50")
    add_ratio("gold_vs_copper", "gcusd_ema_50", "clusd_ema_50")
    add_ratio("nasdaq_vs_copper", "ixic_ema_50", "clusd_ema_50")
    add_ratio("gold_vs_vix", "gcusd_ema_50", "vix_ema_50")
    add_ratio("nasdaq_vs_vix", "ixic_ema_50", "vix_ema_50")
    add_ratio("copper_vs_vix", "clusd_ema_50", "vix_ema_50")
    add_ratio("gold_vs_spread", "gcusd_ema_50", "vix_ema_200")
    add_ratio("vix_vs_yield", "vix_ema_50", "ts_ei_federalFunds_0")

    # 5. Fundamental ratios
    add_ratio("debt_vs_equity", "ts_km_debtToEquity_0", "ts_km_debtToAssets_0")
    add_ratio("debt_vs_roic", "ts_km_debtToEquity_0", "ts_km_roic_0")
    add_ratio("netDebt_vs_roic", "ts_km_netDebtToEBITDA_0", "ts_km_roic_0")
    add_ratio("liquidity_vs_debt", "ts_km_currentRatio_0", "ts_km_debtToAssets_0")
    add_ratio("roe_vs_roic", "ts_km_roe_0", "ts_km_roic_0")
    add_ratio("roe_vs_debt", "ts_km_roe_0", "ts_km_debtToEquity_0")
    add_ratio("roic_vs_assets", "ts_km_roic_0", "ts_km_debtToAssets_0")
    add_ratio("current_vs_debt", "ts_km_currentRatio_0", "ts_km_debtToEquity_0")

    # growth ratios between fundamentals
    add_ratio("roe_growth", "ts_km_roe_0", "ts_km_roe_1")
    add_ratio("roic_growth", "ts_km_roic_0", "ts_km_roic_1")
    add_ratio("debt_growth", "ts_km_debtToEquity_0", "ts_km_debtToEquity_1")

    # 6. Earnings and analyst ratings
    add_ratio(
        "earnings_momentum",
        "ts_earnings_actualEarningResult_0",
        "ts_earnings_actualEarningResult_1",
    )
    add_ratio("rating_momentum", "ts_r_ratingScore_0", "ts_r_ratingScore_1")
    add_ratio(
        "ratingDCF_vs_ROE",
        "ts_r_ratingDetailsDCFScore_0",
        "ts_r_ratingDetailsROEScore_0",
    )
    add_ratio(
        "ratingPE_vs_PB", "ts_r_ratingDetailsPEScore_0", "ts_r_ratingDetailsPBScore_0"
    )
    add_ratio(
        "rating_strength_vs_sentiment",
        "ts_r_ratingScore_0",
        "ts_news_sentiment_company_0",
    )

    add_ratio(
        "buy_vs_sell", "ts_asr_analystRatingsbuy_0", "ts_asr_analystRatingsSell_0"
    )
    add_ratio(
        "strongBuy_vs_strongSell",
        "ts_asr_analystRatingsStrongBuy_0",
        "ts_asr_analystRatingsStrongSell_0",
    )
    add_ratio(
        "hold_vs_buy", "ts_asr_analystRatingsHold_0", "ts_asr_analystRatingsbuy_0"
    )

    # 7. Cross-market and macro sentiment
    add_ratio(
        "sentiment_company_vs_global",
        "ts_news_sentiment_company_0",
        "ts_news_sentiment_global_0",
    )
    add_ratio(
        "sentiment_company_vs_competitors",
        "ts_news_sentiment_company_0",
        "ts_news_sentiment_competitors_0",
    )
    add_ratio("sentiment_vs_vix", "ts_news_sentiment_company_0", "vix_ema_50")
    add_ratio(
        "sentiment_vs_inflation", "ts_news_sentiment_global_0", "ts_ei_inflationRate_0"
    )
    add_ratio("sentiment_vs_roic", "ts_news_sentiment_company_0", "ts_km_roic_0")
    add_ratio("sentiment_vs_rate", "ts_news_sentiment_global_0", "ts_ei_federalFunds_0")
    add_ratio(
        "sentiment_vs_recession",
        "ts_news_sentiment_global_0",
        "ts_ei_smoothedUSRecessionProbabilities_0",
    )

    # 8. Automatic temporal cross ratios
    lag_groups = ["ts_d_close_", "ts_d_volume_", "ts_rsi_14_"]
    for prefix in lag_groups:
        lag_cols = [c for c in df.columns if c.startswith(prefix)]
        for c1, c2 in itertools.combinations(lag_cols[:3], 2):
            df_ratios[f"ratio_{c1}_over_{c2}"] = safe_div(df[c1], df[c2])
    return df_ratios


def main(config=None):

    if config is None:
        import src.config as config

    df_part1A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_X.csv")
    df_part1B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_X.csv")
    df_part2A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_X.csv")
    df_part2B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_X.csv")
    df_part3A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_X.csv")
    df_part3B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_X.csv")

    df_part1A_X = generate_ratios(df_part1A_X)
    df_part1B_X = generate_ratios(df_part1B_X)
    df_part2A_X = generate_ratios(df_part2A_X)
    df_part2B_X = generate_ratios(df_part2B_X)
    df_part3A_X = generate_ratios(df_part3A_X)
    df_part3B_X = generate_ratios(df_part3B_X)

    df_part1A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_X.csv", index=False
    )
    df_part1B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_X.csv", index=False
    )
    df_part2A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_X.csv", index=False
    )
    df_part2B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_X.csv", index=False
    )
    df_part3A_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_X.csv", index=False
    )
    df_part3B_X.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_X.csv", index=False
    )

    df_bench = pd.read_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_benchmark_XY.csv"
    )
    df_bench = generate_ratios(df_bench)
    df_bench.to_csv(
        config.DATA_DIR / f"{config.TRADE_END_DATE}_benchmark_XY.csv", index=False
    )
