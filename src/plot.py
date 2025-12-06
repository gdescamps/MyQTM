"""
plot.py

This module provides utilities for plotting portfolio metrics and comparing them with NASDAQ performance. It generates visualizations and metrics summaries for portfolio analysis.

Functions:
- plot_portfolio_metrics(): Plots portfolio(s) and NASDAQ metrics using the provided metrics dictionary or list of dictionaries.

"""

# %%
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

import src.config as config
from src.interval import get_interval_type


def plot_portfolio_metrics(metrics, nasdaq_metrics=None):
    """
    Plot portfolio(s) and NASDAQ metrics using only the metrics dict or a list of dicts.

    Args:
        metrics (dict or list): Metrics dictionary or list of dictionaries containing portfolio data.
        nasdaq_metrics (dict, optional): Metrics dictionary for NASDAQ performance. Defaults to None.

    Returns:
        tuple: An in-memory PIL image of the plot and a string summary of the metrics.
    """
    # Handle single or multiple metrics
    if isinstance(metrics, list):
        metrics_list = metrics
    else:
        metrics_list = [metrics]

    for m in metrics_list:
        m["nasdaq"] = nasdaq_metrics["nasdaq"]

    # Use NASDAQ from the first metrics dict
    nasdaq_dates_filt = np.array(metrics_list[0]["nasdaq"]["dates_portfolio"])
    nasdaq_values_filt = np.array(metrics_list[0]["nasdaq"]["values_portfolio"])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_yscale("log")

    ymin = None
    ymax = None

    # Plot all portfolios
    # Reverse the metrics_list to plot in reverse order
    for idx, m in enumerate(reversed(metrics_list)):
        dates_portfolio = np.array(m["portfolio"]["dates_portfolio"])
        values_portfolio = np.array(m["portfolio"]["values_portfolio"])

        if len(values_portfolio) == 0:
            continue

        init_capital = values_portfolio[0]
        ymin = init_capital * 0.9
        vmax = np.max(values_portfolio)
        ymax = vmax * 1.1

        if idx == len(metrics_list) - 1:  # first portfolio with blue/red/green
            mask_blue = []
            mask_red = []
            mask_green = []
            mask_orange = []

            for d in dates_portfolio:
                d_str = d.strftime("%Y-%m-%d")
                period = get_interval_type(d_str)
                if "A" in period:
                    mask_red.append(False)
                    mask_blue.append(True)
                    mask_green.append(False)
                    mask_orange.append(False)
                elif "B" in period:
                    mask_red.append(True)
                    mask_blue.append(False)
                    mask_green.append(False)
                    mask_orange.append(False)
                elif "C" in period:
                    mask_red.append(False)
                    mask_green.append(True)
                    mask_blue.append(False)
                    mask_orange.append(False)
                elif "D" in period:
                    mask_red.append(False)
                    mask_green.append(True)
                    mask_blue.append(False)
                    mask_orange.append(True)

            mask_red = np.array(mask_red)
            mask_blue = np.array(mask_blue)
            mask_green = np.array(mask_green)
            mask_orange = np.array(mask_orange)

            ax1.plot(
                dates_portfolio[mask_blue],
                values_portfolio[mask_blue],
                color="blue",
                marker=".",
                linestyle="None",
                markersize=4,
            )
            ax1.plot(
                dates_portfolio[mask_red],
                values_portfolio[mask_red],
                color="red",
                marker=".",
                linestyle="None",
                markersize=4,
            )
            ax1.plot(
                dates_portfolio[mask_green],
                values_portfolio[mask_green],
                color="green",
                marker=".",
                linestyle="None",
                markersize=4,
            )
            ax1.plot(
                dates_portfolio[mask_orange],
                values_portfolio[mask_orange],
                color="orange",
                marker=".",
                linestyle="None",
                markersize=4,
            )

        else:
            # Use a different color for each additional portfolio
            portfolio_colors = [
                "C4",
                "C6",
                "C7",
            ]
            color = portfolio_colors[idx % len(portfolio_colors)]
            ax1.plot(
                dates_portfolio,
                values_portfolio,
                color=color,
                marker=".",
                linestyle="None",
                markersize=2,
            )
    ax1.set_ylim(bottom=ymin, top=ymax)
    ax1.set_ylabel("Portfolio Value ($) / NASDAQ (log scale)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot NASDAQ only once
    ax1.plot(
        nasdaq_dates_filt,
        nasdaq_values_filt,
        color="tab:orange",
        alpha=0.7,
    )

    if config.TRAIN_END_DATE is not None:
        finetune_end_date = pd.to_datetime(config.TRAIN_END_DATE, format="%Y-%m-%d")
        ax1.axvline(
            x=finetune_end_date,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Dual A/B XGBoost models training cut-off date",
        )

    if config.CMAES_END_DATE is not None:
        finetune_end_date = pd.to_datetime(config.CMAES_END_DATE, format="%Y-%m-%d")
        ax1.axvline(
            x=finetune_end_date,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="CMA-ES finetuning cut-off Date",
        )

    # Display metrics for the first portfolio only
    m = metrics_list[0]
    nasdaq_ret = m["nasdaq"]["return"]
    nasdaq_max_drawdown = m["nasdaq"]["max_drawdown"] / 100
    longest_nasdaq_drawdown = m["nasdaq"]["longest_drawdown_period"]

    portfolio_ret_list = []
    portfolio_max_drawdown_list = []
    ulcer_index_list = []
    longest_portfolio_drawdown_list = []
    positions_history_count_list = []
    annual_roi_list = []
    long_rate_list = []
    short_rate_list = []
    AB_rate_list = []
    long_short_rate_list = []
    num_days_list = []

    for m in metrics_list:

        num_days = (dates_portfolio[-1] - dates_portfolio[0]).days
        num_days_list.append(num_days)

        portfolio_ret = m["portfolio"]["return"]
        portfolio_max_drawdown = m["portfolio"]["max_drawdown"] / 100
        ulcer_index = m["portfolio"]["ulcer_index"] / 100
        longest_portfolio_drawdown = m["portfolio"]["longest_drawdown_period"]
        positions_history_count = m["positions_count"]
        dates_portfolio = m["portfolio"]["dates_portfolio"]
        values_portfolio = m["portfolio"]["values_portfolio"]
        annual_roi = m["portfolio"]["annual_roi"]
        long_rate = m["portfolio"]["long_rate"]
        short_rate = m["portfolio"]["short_rate"]
        AB_rate = m["portfolio"]["AB_rate"]
        long_short_rate = m["portfolio"]["long_short_rate"]

        portfolio_ret_list.append(portfolio_ret)
        portfolio_max_drawdown_list.append(portfolio_max_drawdown)
        ulcer_index_list.append(ulcer_index)
        longest_portfolio_drawdown_list.append(longest_portfolio_drawdown)
        positions_history_count_list.append(positions_history_count)
        annual_roi_list.append(annual_roi)
        long_rate_list.append(long_rate)
        short_rate_list.append(short_rate)
        AB_rate_list.append(AB_rate)
        long_short_rate_list.append(long_short_rate)

    portfolio_ret = np.mean(portfolio_ret_list)
    portfolio_max_drawdown = np.mean(portfolio_max_drawdown_list)
    ulcer_index = np.mean(ulcer_index_list)
    longest_portfolio_drawdown = np.mean(longest_portfolio_drawdown_list)
    positions_history_count = np.mean(positions_history_count_list)
    annual_roi = {}
    for ar in annual_roi_list:
        for k, v in ar.items():
            if k not in annual_roi:
                annual_roi[k] = []
            annual_roi[k].append(v)
    for k in annual_roi:
        annual_roi[k] = np.mean(annual_roi[k])
    long_rate = np.mean(long_rate_list)
    short_rate = np.mean(short_rate_list)
    AB_rate = np.mean(AB_rate_list)
    long_short_rate = np.mean(long_short_rate_list)
    num_days = np.mean(num_days_list)

    # Calcul du nombre de jours entre le premier et le dernier trade

    mean_annual_roi = 100.0 * (
        (1 + (portfolio_ret / 100)) ** (1 / (num_days / 365)) - 1
    )
    mean_annual_roi_text = f"Mean annual ROI: {mean_annual_roi:.2f}%\n"

    # Format annual_roi as lines: "YYYY-MM-DD: XX.XX%"
    annual_roi_text = ""
    if annual_roi:
        annual_roi_lines = "\n".join(
            [f"  {k}: {v:.2f}%" for k, v in annual_roi.items()]
        )
        annual_roi_text = f"Annual ROI:\n{annual_roi_lines}\n"

    metrics_text = (
        f"Portfolio:\n"
        f"  Return: {portfolio_ret:.2f}%\n"
        f"  Max DD: {100*portfolio_max_drawdown:.2f}%\n"
        f"  Ulcer Index: {ulcer_index:.2f}\n"
        f"  Pos count: {int(positions_history_count)}\n"
        f"  Longest Drawdown: {int(longest_portfolio_drawdown)}\n"
        f"  Num days: {int(num_days)}\n"
        f"  {mean_annual_roi_text}"
        f"  {annual_roi_text}"
        f"  AB Long Rate: {long_rate:.2f}\n"
        f"  AB Short Rate: {short_rate:.2f}\n"
        f"  AB Rate: {AB_rate:.2f}\n"
        f"  Long/(Long+Short) Rate: {long_short_rate:.2f}\n"
        f"NASDAQ:\n"
        f"  Return: {nasdaq_ret:.2f}%\n"
        f"  Max DD: {100*nasdaq_max_drawdown:.2f}%\n"
        f"  Longest Drawdown: {int(longest_nasdaq_drawdown):.2f}\n"
    )
    ax1.text(
        0.01,
        0.99,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    custom_lines = []
    custom_labels = []

    if config.TRAIN_END_DATE is not None:
        custom_lines.append(Line2D([0], [0], color="red", linestyle="--", linewidth=2))
        custom_labels.append("Train End Date")
    if config.CMAES_END_DATE is not None:
        custom_lines.append(
            Line2D([0], [0], color="orange", linestyle="--", linewidth=2)
        )
        custom_labels.append("Finetune End Date")

    if custom_lines:
        ax1.legend(custom_lines, custom_labels, loc="lower right", fontsize=10)

    plt.title("Portfolio Value, NASDAQ")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image, metrics_text
